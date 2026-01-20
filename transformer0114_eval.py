from transformer0628 import net, src_vocab, tgt_vocab, num_steps, device, tokenize
from d2l_cp import d2l
import torch

net.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))
net = net.to(device)
net.eval()


chs = ['一、属龙的人：天之骄子，乘风破浪']
ens = ['']
for ch, fra in zip(chs, ens):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, ch, src_vocab, tgt_vocab, num_steps, device, True, tokenize)
    print(f'{ch} => {translation}, ')
