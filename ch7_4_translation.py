import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import evaluate
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_dataset
raw_datasets = load_dataset(
    'Helsinki-NLP/kde4', data_dir='en-fr', revision="convert/parquet")
print(raw_datasets)

spilted_datasets = raw_datasets['train'].train_test_split(
    train_size=0.9, seed=20)


print(spilted_datasets['train'][1]['translation'])

max_len = 128

model_checkpoint = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, return_tensors='pt')


def preprocess_funciton(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['fr'] for ex in examples['translation']]
    model_input = tokenizer(inputs, text_target=targets,
                            max_length=max_len, truncation=True)
    return model_input


tokenizer_datasets = spilted_datasets.map(
    preprocess_funciton, batched=True, remove_columns=spilted_datasets['train'].column_names)


# fine_tune
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


metric = evaluate.load('sacrebleu')


def computed_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_specical_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_token=True)

    decoded_preds = [pre.strip() for pre in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    return {
        'bleu': result['score']
    }


tokenizer_datasets.set_format('torch')
train_loader = DataLoader(
    tokenizer_datasets['train'], shuffle=True, collate_fn=data_collator, batch_size=8)

eval_dataloader = DataLoader(
    tokenizer_datasets['test'],  collate_fn=data_collator, batch_size=8)


optimizer = AdamW(model.parameters(), lr=2e-5)


acclerator = Accelerator()
model, optimizer, train_loader, eval_dataloader = acclerator.prepare(
    model, optimizer, train_loader, eval_dataloader)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_loader)
num_train_steps = num_train_epochs * num_update_steps_per_epoch


lr_schedual = get_scheduler('linear', optimizer=optimizer,
                            num_warmup_steps=0, num_training_steps=num_train_epochs)


def post_process(predicitons, labels):
    predicitons = predicitons.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(
        predicitons, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 一些简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[pred.strip()] for pred in decoded_labels]

    return decoded_preds, decoded_labels


process_bar = tqdm(range(num_train_steps))

for epoch in range(num_train_epochs):
    model.train()
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        acclerator.backward(loss)

        optimizer.step()
        lr_schedual.step()
        optimizer.zero_grad()
        process_bar.update(1)

    model.eval()
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            generated_tokens = acclerator.unwrap_model(model).generate(
                batch['input_ids'], attention_mask=batch['attention_mask'], max_length=max_len)

            labels = batch['labels']

            # 预填充
            generated_tokens = acclerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
            labels = acclerator.pad_across_processes(
                labels, dim=1, pad_index=-100)

            predictions_gathered = acclerator.gather(generated_tokens)
            labels_gatherd = acclerator.gather(labels)

            decode_preds, decode_labels = post_process(
                predictions_gathered, labels_gatherd)

            metric.add_batch(predictions=decode_preds,
                             references=decode_labels)
        res = metric.compute()
    print(f'epoch: {epoch}, BLUE score: {res['score']:.2f}')

    acclerator.wait_for_everyone()


print('ok')
