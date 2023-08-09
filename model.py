# imports
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file path for the articles.json file
file_path = 'articles.json'

# load data from the JSON file
with open(file_path) as f:
    data = json.load(f)

# initialize GPT model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# set the padding token
tokenizer.pad_token = tokenizer.eos_token

# create dataset class
class GPTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title_tokens = self.data[idx]['title_tokens']
        content_tokens = self.data[idx]['content_tokens']
        title = ' '.join(title_tokens)
        content = ' '.join(content_tokens)
        text = title + ', ' + content
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze().to(device)
        attention_mask = encoding['attention_mask'].squeeze().to(device)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# maximum length for input sequences
max_length = 512

# create dataset
dataset = GPTDataset(data, tokenizer, max_length)

# create data loader
batch_size = 8
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# set optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# for training
num_epochs = 10
accumulation_steps = 4

for epoch in range(num_epochs):
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # most of this is done to reduce GPU memory usage on google colab
        # bc capitalism
        loss = outputs.loss
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# save fine-tuned model
model_path = 'fine-tuned-model'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)