import math
from tqdm.auto import tqdm
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TrainingArguments
from transformers import default_data_collator
import numpy as np
import collections
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


imdb_dataset = load_dataset("imdb")


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(
            i) for i in range(len(result["input_ids"]))]
    return result


# 使用 batched=True 来激活快速多线程!
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
chunk_size = 128
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15)


def group_texts(examples):
    # 拼接所有的文本
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算拼接文本的长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 如果最后一个块小于 chunk_size,我们将其丢弃
    total_length = (total_length // chunk_size) * chunk_size
    # 按最大长度分块
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # 创建一个新的 labels 列
    result["labels"] = result["input_ids"].copy()
    return result


train_size = 10_000
test_size = int(0.1 * train_size)
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


batch_size = 64
# 在每个 epoch 输出训练的 loss
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

# training_args = TrainingArguments(
#     output_dir=f"{model_name}-finetuned-imdb",
#     overwrite_output_dir=True,
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     weight_decay=0.01,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     push_to_hub=True,
#     fp16=True,
#     logging_steps=logging_steps,
# )


# from transformers import Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=downsampled_dataset["train"],
#     eval_dataset=downsampled_dataset["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )
# trainer.train()

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # 为数据集中的每一列创建一个新的"masked"列
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)


batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # 训练
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 评估
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # 保存并上传
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(output_dir)
    #     repo.push_to_hub(
    #         commit_message=f"Training in progress epoch {epoch}", blocking=False
    #     )
