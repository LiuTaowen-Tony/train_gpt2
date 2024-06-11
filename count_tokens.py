#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load the Wikitext-103 and CNN/DailyMail datasets
wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
cnn_dailymail = load_dataset('cnn_dailymail', '3.0.0', split='train')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Function to count tokens in batches
def count_tokens(dataset, batch_size=1000):
    total_tokens = 0
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_texts = [sample['text'] for sample in batch]
        tokens = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True)['input_ids']
        total_tokens += tokens.numel()
    return total_tokens

# Estimate token count for each dataset
wikitext_token_count = count_tokens(wikitext)
cnn_dailymail_token_count = count_tokens(cnn_dailymail)

# Total token count
total_token_count = wikitext_token_count + cnn_dailymail_token_count

print(f"Wikitext-103 Token Count: {wikitext_token_count}")
print(f"CNN/DailyMail Token Count: {cnn_dailymail_token_count}")
print(f"Total Token Count: {total_token_count}")

