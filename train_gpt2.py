import random
import argparse
import pytorch_lightning as pl
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from torch import nn

import sys
sys.path.append('../..')
import block_linear
from microxcaling.mx import finalize_mx_specs
from microxcaling import mx


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 model from scratch.")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--precision', type=str, default='bf16', help='Precision for training (16 or bf16)')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps')
    # parser.add_argument('--matmul_precision', type=str, default='medium', help="Matrix multiplication precision ('medium' or 'high')")
    parser.add_argument('--token_limit', type=int, default=500_000_000, help="Maximum number of training tokens")
    parser.add_argument('--datasets', nargs='+', default=['wikitext', 'cnn_dailymail'], help="List of datasets to use")
    parser.add_argument('--sampling_probs', nargs='+', type=float, default=None, help="Sampling probabilities for the datasets")
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help="Value for gradient clipping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument('--use_block_float', action='store_true', help='Use block floating point for linear layers')
    return parser.parse_args()

random.seed(42)
torch.manual_seed(42)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Dataset registry
DATASET_REGISTRY = {}

def replace_mx_linear(model, mx_specs):
    def recursive_replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print("replacing", name, "with microxcaling")
                weight = child.weight
                setattr(module, name, mx.Linear(child.in_features, child.out_features, child.bias, mx_specs))
                getattr(module, name).weight = weight
            else:
                recursive_replace_module(child)
    recursive_replace_module(model)
    return model

def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

# Abstract Dataset class
class BaseDataset(IterableDataset):
    def __init__(self, dataset_name, config_name, max_length=128, streaming=False, **kwargs):
        self.dataset = load_dataset(dataset_name, config_name, split='train', streaming=streaming, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for sample in self.dataset:
            yield self.process_sample(sample)

    def process_sample(self, sample):
        raise NotImplementedError("Each dataset must implement its own sample processing method")

@register_dataset('wikitext')
class WikitextDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('wikitext', 'wikitext-103-raw-v1', **kwargs)

    def process_sample(self, sample):
        tokenized = self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids}

@register_dataset('cnn_dailymail')
class CNNDailyMailDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cnn_dailymail', '3.0.0', **kwargs)

    def process_sample(self, sample):
        tokenized = self.tokenizer(sample['article'], padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids}

@register_dataset('c4')
class C4Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('c4', 'en', streaming=True, **kwargs)

    def process_sample(self, sample):
        tokenized = self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids}

class DatasetShuffler(IterableDataset):
    def __init__(self, datasets, token_limit, sampling_probs):
        self.datasets = datasets
        self.token_limit = token_limit
        self.sampling_probs = sampling_probs if sampling_probs else self._calculate_probabilities(datasets)
        self.iterators = [self._recycle_iterator(dataset) for dataset in self.datasets]
        self.current_token_count = 0

    def _calculate_probabilities(self, datasets):
        lengths = [len(dataset.dataset) for dataset in datasets]
        total_length = sum(lengths)
        probabilities = [length / total_length for length in lengths]
        return probabilities

    def _recycle_iterator(self, dataset):
        while True:
            for sample in dataset:
                yield sample

    def __iter__(self):
        while self.current_token_count < self.token_limit:
            chosen_index = random.choices(range(len(self.iterators)), weights=self.sampling_probs, k=1)[0]
            sample = next(self.iterators[chosen_index])
            self.current_token_count += len(sample['input_ids'])
            yield sample

def collate_fn(batch):
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {'input_ids': input_ids, 'labels': labels}

class GPT2FineTuner(pl.LightningModule):
    def __init__(self, model, args, total_tokens):
        super(GPT2FineTuner, self).__init__()
        self.model = model
        self.args = args
        self.total_tokens = total_tokens
        if args.precision == "block_int8":
            self.model = self.model.to(torch.bfloat16)
            block_linear.replace_linear_with_blockwise_int8(self.model)
        elif args.precision == "bf16":
            self.model = self.model.to(torch.bfloat16)
        elif args.precision == "mx_block_int8":
            self.model = self.model.to(torch.bfloat16)
            mx_specs = {
                'scale_bits': 7,
                'w_elem_format': 'int8',
                'a_elem_format': 'int8',
                'mx_block_size': 128,
                'block_size': 128,
                'custom_cuda': True,
                'bfloat': 16,
                # For quantization-aware finetuning, do backward pass in FP32
                'quantize_backprop': True,
            }
            mx_specs = finalize_mx_specs(mx_specs)
            print(mx_specs)
            self.model = replace_mx_linear(self.model, mx_specs)
        else:
            raise ValueError("Precision must be 'block_int8' or 'bf16'")


    def forward(self, input_ids, labels=None):
        return self.model(input_ids, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss.to(torch.bfloat16)
        total_norm = clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_val)
        self.log('grad_norm', total_norm, on_step=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        total_training_steps = self.total_tokens // self.args.batch_size
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=total_training_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def on_after_backward(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, prog_bar=True, logger=True)

# Main training function
def main():
    args = parse_args()

    # torch.set_float32_matmul_precision(args.matmul_precision)

    datasets = [DATASET_REGISTRY[dataset_name](max_length=128) for dataset_name in args.datasets]

    token_limit = args.token_limit
    sampling_probs = args.sampling_probs
    combined_dataset = DatasetShuffler(datasets, token_limit=token_limit, sampling_probs=sampling_probs)
    train_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=256,
        n_layer=6,
        n_head=4,
    )

    model = GPT2LMHeadModel(config)

    #total_tokens = sum(len(sample['input_ids']) for sample in combined_dataset)

    model = GPT2FineTuner(model, args, args.token_limit)

    wandb_logger = WandbLogger(project='gpt2-training')
    wandb_logger.log_hyperparams(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # precision=args.precision,
        logger=wandb_logger,
        limit_train_batches=args.token_limit // (args.batch_size * 128)
    )

    trainer.fit(model, train_dataloader)

    model.model.save_pretrained("./small-gpt2-wikitext-cnn")
    tokenizer.save_pretrained("./small-gpt2-wikitext-cnn")

if __name__ == "__main__":
    main()

