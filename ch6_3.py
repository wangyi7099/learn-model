from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
import torch
example = 'My name is wangyi and I work at School in China'

# tokenier = AutoTokenizer.from_pretrained('bert-base-cased')
# encoding = tokenier(example)

# print(encoding.tokens(), encoding.word_ids(), encoding.word_to_chars(3))


# token_classifier = pipeline("token-classification",
#                             aggregation_strategy="simple")
# res = token_classifier(example)

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)


print(outputs)

print(inputs["input_ids"].shape)
print(outputs.logits.shape)


prob = torch.nn.functional.softmax(outputs.logits, dim=1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)
