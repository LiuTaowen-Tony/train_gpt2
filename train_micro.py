import random
import argparse
import torch.utils
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import torch
import math
import wandb
# import sys
# sys.path.append('../..')
import block_linear
import triton_functions
from microxcaling.mx import finalize_mx_specs
from microxcaling import mx
import os

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 model from scratch.")
    parser.add_argument('--total_batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--precision', type=str, default='bf16', help='Precision for training (16 or bf16)')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--opt_steps', type=int, default=10000, help='Number of warmup steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help="Value for gradient clipping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--quantise_weight', action='store_true', help="Quantise weight for training")
    parser.add_argument('--weight_type', default="fp32")
    parser.add_argument('--fnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--fbnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--bnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--wnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--wbnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--frounding', type=str, default=None)
    parser.add_argument('--brounding', type=str, default=None)
    parser.add_argument('--fbrounding', type=str, default=None)
    parser.add_argument('--wrounding', type=str, default=None)
    parser.add_argument('--wbrounding', type=str, default=None)
    parser.add_argument('--same_input', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--same_weight', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--size', type=float, default=1.0)

    args = parser.parse_args()
    args.per_device_train_batch_size = args.total_batch_size // args.accumulate_grad_batches // torch.cuda.device_count()
    return args

random.seed(42)
torch.manual_seed(42)

def replace_mx_linear(model, mx_specs):
    def recursive_replace_module(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                print(f"replacing {name} with microxcaling")
                weight = child.weight
                setattr(module, name, mx.Linear(child.in_features, child.out_features, child.bias, mx_specs))
                getattr(module, name).weight = weight
            else:
                recursive_replace_module(child)
    recursive_replace_module(model)
    return model

def collate_fn(batch, tokenizer, max_length):
    print(batch)
    inputs = tokenizer(batch, return_tensors='pt', padding='max_length', truncation=True,  max_length=max_length)
    inputs['labels'] = inputs['input_ids'].clone()
    
    return inputs

# def collate_fn(batch, tokenizer, max_length):
#     tokenized = tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_length)
#     input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
#     labels = torch.tensor(tokenized['input_ids'], dtype=torch.long)
#     return {'input_ids': input_ids, 'labels': labels}

def main():
    os.environ["WANDB_PROJECT"] = "train-gpt2" 
    args = parse_args()
    if args.quantise_weight:
        block_linear.QUANTISE_WEIGHT = True

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer

    # Load C4 dataset
    dataset = load_dataset(path='allenai/c4', name='en', split='train', streaming=True, trust_remote_code=True)
    def tokenize_function(examples):
        y = tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        y["label_ids"] = y["input_ids"].clone()
        y["label_ids"][y["label_ids"] == 50256] = -100
        return y
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_train_batch_size)
    # dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')

    # Define model configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=int(256 * math.sqrt(args.size)) // 4 * 4,
        n_layer=int(6 * math.sqrt(args.size)),
        n_head=4,
    )
    model = GPT2LMHeadModel(config)

    # Precision handling
    if args.weight_type == "bf16":
        model = model.to(torch.bfloat16)
    # if args.activation_grad_type == "bf16":
    #     triton_functions.INPUT_OUTPUT_TORCH_TYPE = torch.bfloat16
    if args.precision == "mx_block_int8":
        mx_specs = {
            'scale_bits': 9,
            'w_elem_format': 'int8',
            'a_elem_format': 'int8',
            'mx_block_size': 128,
            'block_size': 128,
            'custom_cuda': True,
            'bfloat': 16,
            'quantize_backprop': True,
        }
        mx_specs = finalize_mx_specs(mx_specs)
        print(mx_specs)
        model = replace_mx_linear(model, mx_specs)
    elif args.precision == "block_int8":
        model = block_linear.replace_linear_with_blockwise_int8(model)
    elif args.precision == "finegrain":


    

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        warmup_steps=args.warmup_steps * args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        bf16=args.precision == 'bf16',
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        max_steps=args.opt_steps * args.accumulate_grad_batches,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer,
                                              padding='max_length',
                                              max_length=128),
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    for i in model.parameters():
        print(i)
        print(i.dtype)
        break

    # Save the model
    model.save_pretrained("./small-gpt2-c4")

if __name__ == "__main__":
    main()
