from datasets import load_dataset
from transformers import AutoTokenizer

data_files = {"train": "train/drugsComTrain_raw.tsv",
              "test": "train/drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")


drug_dataset = drug_dataset.rename_column(
    original_column_name='Unnamed: 0', new_column_name='patient_id')


def lower(example):
    return {'condition': example['condition'].lower()}


def filter(example):
    return example['condition'] is not None


drug_dataset = drug_dataset.filter(filter).map(lower)


def compute_review_length(example):
    return {'review_length': len(example['review'].split())}


drug_dataset = drug_dataset.map(compute_review_length)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # 提取新旧索引之间的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True
)


drug_dataset_clean = tokenized_dataset["train"].train_test_split(
    train_size=0.8, seed=42)
# 将默认的 "test" 部分重命名为 "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 将 "test" 部分添加到我们的 `DatasetDict` 中
drug_dataset_clean["test"] = drug_dataset["test"]


drug_dataset_clean.save_to_disk("drug-reviews")
