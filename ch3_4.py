from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import evaluate

raw_datasets = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenizer_function(example):
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)


tokenized_datasets = raw_datasets.map(tokenizer_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(
    ['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')


train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(
    tokenized_datasets['validation'],  batch_size=8, collate_fn=data_collator)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_train_steps = num_epochs * len(train_dataloader)

lr_schedualer = get_scheduler(
    'linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# tqdm 进度条 了解训练合适结束

processbar = tqdm(range(num_train_steps))
model.train()


for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        loss = output.loss
        loss.backward()

        optimizer.step()
        lr_schedualer.step()
        optimizer.zero_grad()
        processbar.update(1)


metric = evaluate.load('glue', 'mrpc')
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)

    logits = output.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch['labels'])

metric.compute()
