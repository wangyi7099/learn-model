from tqdm.auto import tqdm
import torch
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import evaluate
from transformers import AutoTokenizer
from datasets import concatenate_datasets, DatasetDict
from datasets import load_dataset

dataset_en = load_dataset(
    'json', data_files=r"D:\kf\data\reviews\en\train.jsonl")
dataset_zh = load_dataset(
    'json', data_files=r"D:\kf\data\reviews\zh\train.jsonl")

dataset_zh = dataset_zh['train'].train_test_split(train_size=0.8)
dataset_en = dataset_en['train'].train_test_split(train_size=0.8)
print(dataset_zh.num_columns)


dataset_en.set_format("pandas")
english_df = dataset_en["train"][:]
print(english_df["product_category"].value_counts()[:20])
dataset_en.reset_format()


def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset["train"].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Title: {example['review_title']}'")
        print(f"'>> Review: {example['review_body']}'")


def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )


chinese_books = dataset_zh.filter(filter_books)
english_books = dataset_en.filter(filter_books)


show_samples(chinese_books)


book_datasets = DatasetDict()
for split in english_books.keys():
    book_datasets[split] = concatenate_datasets([
        english_books[split], chinese_books[split]
    ])
    book_datasets[split] = book_datasets[split].shuffle(seed=42)

test_dataset = book_datasets['test']
test_dataset = test_dataset.train_test_split(1000)
book_datasets['test'] = test_dataset['train']
book_datasets['validation'] = test_dataset['test']

show_samples(book_datasets)


book_datasets = book_datasets.filter(
    lambda x: len((x['review_title'] or '').split()) > 2)


model_checkpoint = 'google/mt5-small'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# inputs = tokenizer('你好')

# print(inputs)

max_input_len = 512
max_target_len = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['review_body'],
        max_length=max_input_len,
        truncation=True
    )

    labels = tokenizer(
        examples['review_title'],
        max_length=max_target_len,
        truncation=True
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_datasets = book_datasets.map(preprocess_function, batched=True)


rouge_score = evaluate.load('rouge')


nltk.download('punkt')
nltk.download('punkt_tab')


def three_sentence_summary(text):
    return '\n'.join(sent_tokenize(text)[:3])


def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text)
                 for text in dataset['review_body']]
    return metric.compute(predictions=summaries, references=dataset['review_title'])


score = evaluate_baseline(book_datasets['validation'], rouge_score)
rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
rouge_dict = dict(
    (rn, round(score[rn] * 100, 2)) for rn in rouge_names)

print(rouge_dict)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# T8D7XIBDUPTJL3UR

batch_size = 8
num_train_epochs = 8
logging_steps = len(book_datasets['train']) // batch_size
model_name = model_checkpoint.split('/')[-1]


# args = Seq2SeqTrainingArguments(
#     output_dir=model_name,
#     eval_strategy='epoch',
#     learning_rate=5.6e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     weight_decay=1e-2,
#     save_total_limit=3,
#     num_train_epochs=num_train_epochs,
#     predict_with_generate=True,
#     logging_steps=logging_steps,
#     push_to_hub=False,
#     save_safetensors=True
# )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    predictions = np.where(predictions != -100,
                           predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip()))
                      for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


tokenized_datasets = tokenized_datasets.remove_columns(
    book_datasets["train"].column_names
)


# trainer = Seq2SeqTrainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_datasets['train'],
#     eval_dataset=tokenized_datasets['validation'],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()
# trainer.evaluate()


tokenized_datasets.set_format("torch")


batch_size = 8
train_dataloader = DataLoader(
    tokenized_datasets['train'],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size
)

eval_dataloader = DataLoader(
    tokenized_datasets['validation'], collate_fn=data_collator, batch_size=batch_size
)

optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()


model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)


num_train_epochs = 10
num_update_steps_per_epoch = len(train_dataloader)
num_train_steps = num_train_epochs * num_update_steps_per_epoch

lr_schedual = get_scheduler('linear', optimizer=optimizer,
                            num_warmup_steps=0, num_training_steps=num_train_epochs)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [pred.strip() for pred in labels]

    preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ['\n'.join(nltk.sent_tokenize(pred)) for pred in labels]

    return preds, labels


process_bar = tqdm(range(num_train_steps))


for epoch in range(num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_schedual.step()
        optimizer.zero_grad()
        process_bar.update(1)

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_token = accelerator.unwrap_model(model).generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            generated_token = accelerator.pad_across_processes(
                generated_token, dim=1, pad_index=tokenizer.pad_token_id
            )

            labels = batch["labels"]

            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_token, tuple):
                generated_token = generated_token[0]
            decoded_preds = tokenizer.batch_decode(
                generated_token, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(
                decoded_preds, decoded_preds
            )

            rouge_score.add_batch(predictions=decoded_preds,
                                  references=decoded_labels)

        result = rouge_score.compute()
        result = {key: value * 100 for key, value in result.items()}

        print(f'Epoch {epoch}:', result)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_name)
