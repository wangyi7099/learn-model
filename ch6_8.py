from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]


# tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# tokenizer.normalizer = normalizers.Sequence(
#     [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
# )
# tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# pre_tokenizer = pre_tokenizers.Sequence(
#     [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
# )
# pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer.")
# special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# trainer = trainers.WordPieceTrainer(
#     vocab_size=25000, special_tokens=special_tokens)

# tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# cls_token_id = tokenizer.token_to_id("[CLS]")
# sep_token_id = tokenizer.token_to_id("[SEP]")
# tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"[CLS]:0 $A:0 [SEP]:0",
#     pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
#     special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
# )
# encoding = tokenizer.encode("Let's test this tokenizer.")
# print(encoding.tokens)

# GPT2
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
print('prev:\n')
print(tokenizer.pre_tokenizer.pre_tokenize_str('let\'s test pre-tokenization!'))
trainer = trainers.BpeTrainer(
    vocab_size=25000, special_tokens=['<|endoftext|>'])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
encoding = tokenizer.encode('let\'s test pre-tokenization!')
print('after:\n')
print(encoding.tokens)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
sentence = "Let's test this tokenizer."
print('new:\n')
encoding = tokenizer.encode(sentence)
print(encoding.tokens)
start, end = encoding.offsets[4]
print(sentence[start:end])
tokenizer.decoder = decoders.ByteLevel()
print('decode:', ','.join(tokenizer.decode(encoding.ids)))
