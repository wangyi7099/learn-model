import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# 这一行会运行失败
model(input_ids)
'''
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

seq = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(seq)
ids = tokenizer.convert_tokens_to_ids(tokens)
inputs_ids = torch.tensor([ids])

output = model(inputs_ids)
print(output.logits)
