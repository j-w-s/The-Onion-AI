# imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'fine-tuned-model'
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

title_prompt = str(input('Enter title prompt here: '))
title_input_ids = tokenizer.encode(title_prompt, return_tensors='pt').to(device)
max_title_length = 100
title_output = model.generate(title_input_ids, max_length=max_title_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_title = tokenizer.decode(title_output[0], skip_special_tokens=True)

body_prompt = str(input('Enter body prompt here: '))
combined_prompt = generated_title + ', ' + body_prompt
combined_input_ids = tokenizer.encode(combined_prompt, return_tensors='pt').to(device)
max_body_length = 400
body_output = model.generate(combined_input_ids, max_length=max_body_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
generated_body = tokenizer.decode(body_output[0], skip_special_tokens=True)

print("Generated Title:", generated_title)
print("Generated Body:", generated_body)
