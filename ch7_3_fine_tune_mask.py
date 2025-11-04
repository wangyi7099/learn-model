import math
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TrainingArguments, default_data_collator, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import collections
from datasets import load_dataset
import torch
from transformers import get_scheduler

model_checkpoint = 'distilbert-base-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

imdb_dataset = load_dataset('imdb')


def tokenizer_function(examples):
    result = tokenizer(examples['text'])
    if tokenizer.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))
                              ]
    return result


tokenizer_datasets = imdb_dataset.map(
    tokenizer_function, batched=True, remove_columns=['text', 'label'])
chunk_size = 128
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result


train_size = 10_000
test_size = int(0.1*train_size)
lm_datasets = tokenizer_datasets.map(group_texts, batched=True)
downsampled_dataset = lm_datasets['train'].train_test_split(
    train_size=train_size,
    test_size=test_size,
    seed=42
)
batch_size = 64
logging_steps = len(downsampled_dataset['train']) // batch_size
model_name = model_checkpoint.split('/')[-1]


def insert_random_mask(batch):
    feats = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(feats)
    return {'masked_' + k: v.numpy() for k, v in masked_inputs.items()}


downsampled_dataset = downsampled_dataset.remove_columns(['word_ids'])
eval_dataset = downsampled_dataset['test'].map(
    insert_random_mask, batched=True, remove_columns=downsampled_dataset['test'].column_names)

eval_dataset = eval_dataset.rename_columns({
    "masked_input_ids": "input_ids",
    "masked_attention_mask": "attention_mask",
    "masked_labels": "labels",
})

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset['train'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator
)
eval_data_loader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
optimizer = AdamW(model.parameters(), lr=5e-5)
accelerator = Accelerator()

model, optimizer, train_dataloader, eval_data_loader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_data_loader
)


num_train_epochs = 3
num_update_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_per_epoch


lr_scheduler = get_scheduler('linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=num_training_steps)


progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    model.eval()
    losses = []
    for step, batch in enumerate(eval_data_loader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float('inf')
    print(f'>>> Epoch {epoch}: Perplexity: {perplexity}')
