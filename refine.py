# imports
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import GPTDataset

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file path for the articles.json file
file_path = 'articles.json'

# load data from the JSON file
with open(file_path) as f:
    data = json.load(f)

initial_model_path = 'fine-tuned-model'

model = GPT2LMHeadModel.from_pretrained(initial_model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(initial_model_path)

validation_dataset = GPTDataset(data, tokenizer, max_length)
validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
max_iterations = 50
best_perplexity = float('inf')  # high value for overriding original model during training

for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}")

    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0, device=device)

    with torch.no_grad():
        for batch in validation_data_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            total_loss = torch.add(total_loss, torch.mul(loss.item(), attention_mask.sum().to(torch.float)))
            total_tokens = torch.add(total_tokens, attention_mask.sum())

    # lower is better
    current_perplexity = torch.exp(torch.div(total_loss, total_tokens))

    if current_perplexity < best_perplexity:
        best_perplexity = current_perplexity
        refined_model_path = 'fine-tuned-model'
        model.save_pretrained(refined_model_path)
        tokenizer.save_pretrained(refined_model_path)

        print(f"Saved refined model with improved validation perplexity: {refined_model_path}")