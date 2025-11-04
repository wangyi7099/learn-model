from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForTokenClassification
import torch

# context = '''
# Jack's full name is K Jack, he is 16 years old, he like basketball and eating banana.
# '''
# question = 'What is Jack\'s full name?'


# model_checkpoint = "distilbert-base-cased-distilled-squad"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# inputs = tokenizer(question, context, return_tensors="pt")
# outputs = model(**inputs)


# sequence_ids = inputs.sequence_ids()
# # 屏蔽除 context 之外的所有内容
# mask = [i != 1 for i in sequence_ids]
# # 不屏蔽 [CLS] token
# mask[0] = False
# mask = torch.tensor(mask)[None]


# start_logits = outputs.start_logits
# end_logits = outputs.end_logits


# start_logits[mask] = -1e5
# end_logits[mask] = -1e5

# start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
# end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

# s_start = start_probabilities[:, None]
# s_end = end_probabilities[None, :]


# scores = s_start * s_end
# scores = torch.triu(scores)


# max_index = scores.argmax().item()

# start_index = max_index // scores.shape[1]
# end_index = max_index % scores.shape[1]

# inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
# offsets = inputs_with_offsets['offset_mapping']


# start_char, _ = offsets[start_index]
# _, end_char = offsets[end_index]
# answer = context[start_char:end_char]


# print('answer:', answer)


# sentence = "This sentence is not too long but we are going to split it anyway."
# inputs = tokenizer(
#     sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
# )

# for ids in inputs["input_ids"]:
#     print(tokenizer.decode(ids))


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
    "Hello, how are  you?"))
