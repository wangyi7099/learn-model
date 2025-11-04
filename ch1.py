from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

checkpoint = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
model_clss = AutoModelForSequenceClassification.from_pretrained(checkpoint)
raw_inputs = [
    'h e l l'
]

inputs = tokenizer(raw_inputs, padding=True,
                   truncation=True, return_tensors='pt')


outputs = model_clss(**inputs)

# 1. 拿到真正的 token 字符串
tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])   # 返回 List[str]
print("输入 token 序列:", tokens)

# 2. 如果想同时看 id
for id_, tok in zip(inputs.input_ids[0], tokens):
    print(f"id {id_:>6} -> {tok}")
