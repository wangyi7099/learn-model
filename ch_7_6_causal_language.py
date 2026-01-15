from tqdm.notebook import tqdm
from transformers import get_scheduler
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict


def any_keywords_in_strings(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False


filters = ["pandas", "sklearn", "matplotlib", "seaborn"]


# def filter_streaming_dataset(dataset, filters):
#     filtered_dict = defaultdict(list)
#     total = 0
#     for example in tqdm(iter(dataset)):
#         total += 1
#         if any_keywords_in_strings(example, filters):
#             for k, v in example.items():
#                 filtered_dict[k].append(v)

#     print(
#         f'{len(filtered_dict['content']) / total:.2%} of data after filtering')

#     return Dataset.from_dict(filtered_dict)


# split = 'train'
# filters = ['pandas', 'sklearn', 'matplotlib', 'seaborn']

# dataset = load_dataset(
#     f"transformersbook/codeparrot-{split}", split=split, revision="convert/parquet", streaming=True)
# filtered_dataset = filter_streaming_dataset(dataset, filters)


ds_train = load_dataset(
    "huggingface-course/codeparrot-ds-train", split="train", revision="convert/parquet")
ds_valid = load_dataset(
    "huggingface-course/codeparrot-ds-valid", split="validation", revision="convert/parquet")

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)


context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
    'huggingface-course/code-search-net-tokenizer')


'''
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
tokenized_datasets
'''


def tokenize(element):
    outputs = tokenizer(
        element['content'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True
    )

    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length == context_length:
            input_batch.append(input_ids)
        return {'input_ids': input_batch}


tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets['train'].column_names
)
config = AutoConfig.from_pretrained(
    'gpt2',
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
model = GPT2LMHeadModel(config)


tokenizer.pad_token = tokenizer.eos_token
data_collactor = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False)


args = TrainingArguments(
    output_dir='codeparrot-ds',
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    eval_strategy='steps',
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1000,
    lr_scheduler_type='cosine',
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collactor,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid']
)

trainer.train()


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    loss_fct = CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    loss_per_sample = loss.view(shift_logits.size(
        0), shift_logits.size(1)).mean(axis=1)
    weights = torch.stack([(inputs == kt).float()
                          for kt in keytoken_ids]).sum(axis=[0, 2])
    weights = alpha * (1.0 + weights)
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss


tokenized_datasets.set_format('torch')

train_dataloader = DataLoader(
    tokenized_datasets['train'], batch_size=32, shuffle=True)

eval_dataloader = DataLoader(
    tokenized_datasets['valid'], batch_size=32)


weight_decay = 0.1


def get_groupded_params(model, no_decay=['bias', 'LayerNorm.weight']):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)

    return [
        {'params': params_with_wd, 'weight_decay': weight_decay},
        {'params': params_without_wd, 'weight_decay': 0},
    ]


def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch['input_ids'], labels=batch['input_ids'])
        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float('inf')

    return loss.item(), perplexity.item()


model = GPT2LMHeadModel(config)


optimizer = AdamW(get_groupded_params(model), lr=5e-4)


accelerator = Accelerator(fp16=True)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)


num_train_epoches = 1
num_update_steps_per_epochs = len(train_dataloader)
num_train_steps = num_train_epoches * num_update_steps_per_epochs

lr_schedual = get_scheduler(
    name='linear',
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_train_steps
)

keytoken_ids = []
for keyword in [
    "plt",
    "pd",
    "sk",
    "fit",
    "predict",
    " plt",
    " pd",
    " sk",
    " fit",
    " predict",
    "testtest",
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"Keyword has not single token: {keyword}")


gradient_accumulation = 8
eval_step = 5_000

model.train()
completed_steps = 0

for epoch in range(num_train_epoches):
    for step, batch in tqdm(enumerate(train_dataloader), total=num_train_steps):
        logits = model(batch['inputs_ids']).logits
        loss = keytoken_weighted_loss(batch['input_ids'], logits, keytoken_ids)
        if step % 100 == 0:
            accelerator.print({
                'samples': step * len(batch),
                'steps': completed_steps,
                'loss/train': loss.item() * gradient_accumulation
            })
        loss = loss / gradient_accumulation
        accelerator.backward(loss)
