from transformer102 import Transformer, D12_L2, D12_featurize, D12_FC_out, D12_linear, EncoderLayer, LayerNorm_invariant, PositionalEncoding
import pickle
import math
import torch
from torch import nn

irrep_list = ["A1", "B2", "E1", "E2", "E3", "E4", "E5"]

mult_input = {k: 1 for k in irrep_list}
mult_model = {k: 64 for k in irrep_list}

with open("D12_Q.pkl", "rb") as f:
    D12_Q = pickle.load(f)
for k in ["E1", "E2", "E3", "E4", "E5"]:
    D12_Q[k] *= math.sqrt(2)

tf = Transformer(mult_input, mult_model, 8, 4, mult_model, 0, D12_Q)

c0 = torch.tensor([[[1,2,1,3,1,4,1,5,1,6,1,7], [1,2,1,3,1,4,1,5,1,6,1,7], [1,2,1,3,1,4,1,5,1,6,1,7]]], dtype=torch.float32)
# c1 = torch.tensor([[[7,1,2,1,3,1,4,1,5,1,6,1], [7,1,2,1,3,1,4,1,5,1,6,1], [7,1,2,1,3,1,4,1,5,1,6,1]]], dtype=torch.float32)
c1 = torch.tensor([[[1,7,1,6,1,5,1,4,1,3,1,2], [1,7,1,6,1,5,1,4,1,3,1,2], [1,7,1,6,1,5,1,4,1,3,1,2]]], dtype=torch.float32)
    
f = D12_featurize(D12_Q)
p = PositionalEncoding(mult_model)
l = D12_linear(mult_input, mult_model)
en = EncoderLayer(mult_model, 8, mult_model,0)
ln = LayerNorm_invariant(mult_model)
o = D12_FC_out(mult_model, D12_Q)

# print(o(en(p(l(f(c0))))))
# print(o(en(p(l(f(c1))))))
# print(o(en(en(p(l(f(c0)))))))
# print(o(en(en(p(l(f(c1)))))))
# print(o(ln(p(l(f(c0))))) * 100)
# print(o(ln(p(l(f(c1))))) * 100)
# print(o(en(en(en(p(l(f(c0))))))))
# print(o(en(en(en(p(l(f(c1))))))))
print(tf(c0))
print(tf(c1))