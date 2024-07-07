import numpy as np
import random
import argparse
import torch.utils
from transformers import (
    MarianTokenizer, MarianMTModel, MarianConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator
)
from datasets import load_dataset, load_metric
import qtorch
import torch
import math
import wandb
from itertools import chain
import torchsummary


# import block_linear
# import triton_functions
# from microxcaling.mx import finalize_mx_specs
# from microxcaling import mx

from low_precision_utils import utils

import os

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 model from scratch.")
    parser.add_argument('--total_batch_size', type=int, default=80, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--precision', type=str, default='bf16', help='Precision for training (16 or bf16)')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--opt_steps', type=int, default=100_000, help='Number of warmup steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help="Value for gradient clipping")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--quantise_weight', action='store_true', help="Quantise weight for training")
    parser.add_argument('--weight_type', default="fp32")
    parser.add_argument('--fnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--bfnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--bnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--wnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--bwnumber', type=int, default=None, help="Number of bits for the fraction part")
    parser.add_argument('--frounding', type=str, default=None)
    parser.add_argument('--brounding', type=str, default=None)
    parser.add_argument('--bfrounding', type=str, default=None)
    parser.add_argument('--wrounding', type=str, default=None)
    parser.add_argument('--bwrounding', type=str, default=None)
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

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

def main():
    os.environ["WANDB_PROJECT"] = "wmt-14-en-de-precision" 
    args = parse_args()
    if args.quantise_weight:
        block_linear.QUANTISE_WEIGHT = True

    # Load tokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    # tokenizer.pad_token = tokenizer.eos_token

    # Load C4 dataset
    # dataset = load_dataset(path='allenai/c4', name='en', split='train', streaming=True, trust_remote_code=True)
    dataset = load_dataset('wmt14', 'de-en')

    def preprocess_function(examples):
        source_lang = 'en'
        target_lang = 'de'
        inputs = [example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    metric = load_metric("sacrebleu", trust_remote_code=True)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print(preds, labels)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # def tokenize_function(examples):
    #     y = tokenizer(examples['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    #     y["label_ids"] = y["input_ids"].clone()
    #     y["label_ids"][y["label_ids"] == 50256] = -100
    #     return y
    # dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


    # print(dataset._head())
    config = MarianConfig(
        vocab_size=tokenizer.vocab_size,  # Use the vocab size from the tokenizer
        max_position_embeddings=512,
        encoder_layers=6,
        encoder_ffn_dim=2048,
        encoder_attention_heads=8,
        decoder_layers=6,
        decoder_ffn_dim=2048,
        decoder_attention_heads=8,
        d_model=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        classifier_dropout=0.1,
        use_cache=True,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )

    # Initialize the MarianMT model with the defined configuration
    model = MarianMTModel(config)
    # fake_input = torch.randint(0, 1000, (128, 128))
    # print(torchsummary.summary(model, (1, 128, 128), dtype=torch.long, device="cpu"))
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)




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
        build_number = lambda x : qtorch.FloatingPoint(8, x)
        quant_scheme = utils.QuantScheme(
            fnumber=build_number(args.fnumber),
            bnumber=build_number(args.bnumber),
            wnumber=build_number(args.wnumber),
            fround_mode=args.frounding,
            bround_mode=args.brounding,
            wround_mode=args.wrounding,
            same_input=args.same_input,
            same_weight=args.same_weight,
            bfnumber=build_number(args.bfnumber),
            bwnumber=build_number(args.bwnumber),
            bfround_mode=args.bfrounding,
            bwround_mode=args.bwrounding
        )
        print(quant_scheme)
        # model = utils.QuantWrapper(model, quant_scheme)
        model = utils.replace_with_quantized(model, quant_scheme)
        

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='/vol/bitbucket/tl2020/translation-results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        per_device_eval_batch_size=64,
        warmup_steps=args.warmup_steps * args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",
        gradient_checkpointing=True,
        bf16=args.precision == 'bf16',
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        eval_steps=1000,
        eval_strategy="steps",
        # max_steps=args.opt_steps * args.accumulate_grad_batches,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train the model
    trainer.train()
    # model.save_pretrained("./small-gpt2-c4")

if __name__ == "__main__":
    main()
