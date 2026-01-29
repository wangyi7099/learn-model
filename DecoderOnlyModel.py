from transformers.processing_utils import Unpack
from transformers.integrations import use_kernel_forward_from_hub
import jieba
import d2l_cp as d2l
import math
import pandas as pd
import torch
from torch import nn
import matplotlib
from transformers import PreTrainedModel, GenerationMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from typing import Callable, Optional, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
matplotlib.use('Agg')  # 设置非交互式后端


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class DecoderBlock(nn.Module):
    """解码器第i个块"""

    def __init__(self, num_hidden, normal_shape, ffn_input, ffn_hidden,
                 num_heads, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            num_hidden, num_heads, dropout)
        self.addnorm1 = AddNorm(normal_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            num_hidden, num_heads, dropout)
        self.addnorm2 = AddNorm(normal_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input, ffn_hidden, num_hidden)

        self.addnormal3 = AddNorm(normal_shape, dropout)

    def forward(self, X, state):
        enc_outpus, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:  # ?????start
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头(batch_size, num_steps),
            # 其中每一行是[1,2,...,num_steps]
            # 注意不是enc_valid_lens
            dec_valid_lens = torch.arange(
                1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 掩蔽自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 编码-解码器注意力
        # end_outputs的开头  (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outpus, enc_outpus, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnormal3(Z, self.ffn(Z)), state


class DecoderOnlyModelConfig(PretrainedConfig):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, rms_norm_eps, **kwargs):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.norm_shape = norm_shape
        self.ffn_num_input = ffn_num_input
        self.ffn_num_hiddens = ffn_num_hiddens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.rms_norm_eps = rms_norm_eps
        super().__init__(
            **kwargs,
        )
        pass


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class QWEN2RotaryEmbedding(nn.Module):
    def __init__(self, config: DecoderOnlyModelConfig, device=None):
        super().__init__()


class DecoderModel(PreTrainedModel):
    def __init__(self, config: DecoderOnlyModelConfig):
        super().__init__(config)
        self.num_hiddens = config.num_hiddens
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(
            config.num_hiddens, config.dropout)
        self.blks = nn.Sequential()
        self.norm = Qwen2RMSNorm(config.num_hiddens, eps=config.rms_norm_eps)
        self.rotary_emb = QWEN2RotaryEmbedding(config=config)
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(config.num_hiddens, config.norm_shape, config.ffn_num_input, config.ffn_num_hiddens,
                                              config.num_heads, config.dropout, i))
        self.dense = nn.Linear(config.num_hiddens, config.vocab_size)

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class DecoderOnlyModelDecoder(PreTrainedModel, GenerationMixin):
    # def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    #              num_heads, num_layers, dropout, **kwargs):
    def __init__(self, config: DecoderOnlyModelConfig):
        super().__init__(config)
        self.model = DecoderModel(config)
        self.lm_head = nn.Linear(
            config.num_hiddens, config.vocab_size, bias=False)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep,
                              None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 50
lr, num_epochs, device = 0.005, 1000, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
