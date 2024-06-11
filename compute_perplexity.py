import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Custom Dataset class to handle text sequences
class TextDataset(Dataset):
      def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                        self.tokenizer = tokenizer
                        self.max_length = max_length

                    def __len__(self):
                              return len(self.texts)

                                  def __getitem__(self, idx):
                                            text = self.texts[idx]
                                                    inputs = self.tokenizer.encode_plus(
                                                                    text,
                                                                                max_length=self.max_length,
                                                                                            truncation=True,
                                                                                                        padding='max_length',
                                                                                                                    return_tensors='pt'
                                                                                                                            )
                                                    input_ids = inputs['input_ids'].squeeze()
                                                    attention_mask = inputs['attention_mask'].squeeze()
                                                    return input_ids, attention_mask

                                            def compute_perplexity(model, dataloader, device):
                                                  model.eval()
                                                      total_log_likelihood = 0.0
                                                      total_tokens = 0

                                                      with torch.no_grad():
                                                                for batch in dataloader:
                                                                            input_ids, attention_mask = [x.to(device) for x in batch]
                                                                            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                                                                            log_likelihood = outputs.loss.item() * input_ids.size(0)  # Cross entropy loss is per token
                                                                            total_log_likelihood += log_likelihood
                                                                            total_tokens += input_ids.size(0) * input_ids.size(1)

                                                                    avg_log_likelihood = total_log_likelihood / total_tokens
                                                                        perplexity = np.exp(avg_log_likelihood)
      return perplexity

      def main():
            # Sample data
            texts = [
                    "This is a sample sentence.",
                    "Another example of a sentence.",
                    "The quick brown fox jumps over the lazy dog."
                ]
                    
                    # Parameters
                    model_name = 'gpt2'
                        max_length = 50
                            batch_size = 2
                                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                    # Load model and tokenizer
                                    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
      model = GPT2LMHeadModel.from_pretrained(model_name)
      model.to(device)

      # Prepare dataset and dataloader
      dataset = TextDataset(texts, tokenizer, max_length)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

      # Compute perplexity
      perplexity = compute_perplexity(model, dataloader, device)
      print(f'Perplexity: {perplexity}')

      if __name__ == '__main__':
          main()

