from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import notebook_login

raw_datasets = load_dataset("Nan-Do/code-search-net-python")
print(raw_datasets["train"][123456]["code"])


def get_train_corpus():
    return (
        raw_datasets['train'][i:i+1000]['code']
        for i in range(0, len(raw_datasets['train']), 1000)
    )


# old_tokenizer = AutoTokenizer.from_pretrained('gpt2')


# training_corpus = get_train_corpus()
# tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokenizer = AutoTokenizer.from_pretrained('code-search-net-tokenizer')


tokenizer.push_to_hub("code-search-net-tokenizer")
