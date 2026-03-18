from typing import List, Optional, Union
import json
import os
from transformers import PreTrainedTokenizer
from transformers.utils import PaddingStrategy


class MyCustomTokenizer(PreTrainedTokenizer):
    """
    自定义 Tokenizer 示例
    - vocab: 简单的字符级 + 常用词表
    - 支持 save_pretrained / from_pretrained
    """

    # 定义 special tokens
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        **kwargs
    ):
        # 必须先设置 special tokens，再调父类 __init__
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # 加载或创建词表
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            # 默认最小词表（实际应从数据训练 BPE/SentencePiece）
            self.vocab = self._build_default_vocab()

        # id -> token 反向映射
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )

    def _build_default_vocab(self) -> dict:
        """构建默认词表：数字 + 小写字母 + 常用符号 + 预留词位"""
        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        idx = 4

        # 数字
        for i in range(10):
            vocab[str(i)] = idx
            idx += 1

        # 小写字母
        for c in "abcdefghijklmnopqrstuvwxyz":
            vocab[c] = idx
            idx += 1

        # 常用符号
        for c in " .,!?;:'\"-()[]{}":
            vocab[c] = idx
            idx += 1

        # 预留 100 个常用词位（实际应通过 BPE 训练）
        common_words = ["the", "and", "is", "are",
                        "this", "that", "with", "for", "to", "of"]
        for word in common_words:
            vocab[word] = idx
            idx += 1

        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict:
        return self.vocab.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        核心分词逻辑：这里用极简的空格+字符回退策略
        生产环境应使用 BPE/SentencePiece/Unigram 算法
        """
        text = text.lower().strip()
        tokens = []
        i = 0

        while i < len(text):
            # 尝试匹配最长词
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):  # 最长匹配20字符
                substr = text[i:i+length]
                if substr in self.vocab and substr not in [self.unk_token, self.pad_token]:
                    tokens.append(substr)
                    i += length
                    matched = True
                    break

            if not matched:
                # 回退到字符级
                char = text[i]
                tokens.append(char if char in self.vocab else self.unk_token)
                i += 1

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将 token 列表还原为字符串"""
        return "".join(tokens).replace("▁", " ")  # 如果有 SentencePiece 风格空格标记

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """添加 BOS/EOS 等特殊 token"""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 句子对：BOS A EOS B EOS
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """保存词表到文件，供 save_pretrained 调用"""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") +
            self.vocab_files_names["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)


# ============ 使用示例 ============

if __name__ == "__main__":
    # 1. 创建 tokenizer
    tokenizer = MyCustomTokenizer()

    # 2. 测试编码解码
    text = "the cat is 123"
    encoded = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"IDs: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # 3. 批量编码
    batch = tokenizer(
        ["the cat", "hello world 42"],
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )
    print(f"\nBatch: {batch}")

    # 4. 保存和加载
    tokenizer.save_pretrained("./my_tokenizer")

    # 从保存的路径加载
    loaded_tokenizer = MyCustomTokenizer.from_pretrained("./my_tokenizer")
    print(f"\nLoaded vocab size: {loaded_tokenizer.vocab_size}")
