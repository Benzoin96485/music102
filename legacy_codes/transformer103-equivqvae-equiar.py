import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os import remove, chdir
import pickle

with open("D12_Q.pkl", "rb") as f:
    D12_Q = pickle.load(f)
for k in ["E1", "E2", "E3", "E4", "E5"]:
    D12_Q[k] *= math.sqrt(2)


class D12_featurize(nn.Module):
    def __init__(self, D12_Q, dtype=torch.float32):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.ones(1))
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=dtype), requires_grad=False) for k, v in D12_Q.items()
        })
    
    def forward(self, D12_vec_perm):
        return {
            k: v @ (D12_vec_perm + self.bias).unsqueeze(-1) * self.weight for k, v in self.Q.items()
        }

# class D12_linear(nn.Module):
#     def __init__(self, input_mult, output_mult):
#         super().__init__()
#         self.din = sum(input_mult.values())
#         self.dout = sum(output_mult.values())
#         self.linear = nn.Linear(self.din, self.dout)
#         self.Q = nn.ParameterDict({
#             k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False) for k, v in D12_Q.items()
#         })
#         self.keys = ["A1", "B2", "E1", "E2", "E3", "E4", "E5"]
#         self.splits = [output_mult[k] for k in self.keys]

#     def forward(self, D12_vec):
#         out = dict()
#         D12_perm = torch.concat([self.Q[k].T @ D12_vec[k] for k in self.keys], dim=-1)
#         D12_perm = self.linear(D12_perm)
#         D12_perm_split = torch.split(D12_perm, self.splits, dim=-1)
#         for i, k in enumerate(self.keys):
#             out[k] = self.Q[k] @ D12_perm_split[i]
#         return out

class D12_linear_single(nn.Module):
    def __init__(self, input_mult, output_mult):
        super().__init__()
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.linear_channels = nn.ModuleDict({
            k: nn.Linear(v, output_mult[k], bias=False) for k, v in input_mult.items()
        })
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False) for k, v in D12_Q.items()
        })

    def forward(self, D12_vec):
        return {
            k: self.linear_channels[k](v) for k, v in D12_vec.items()
        }

D12_linear = D12_linear_single

def D12_L2(D12_vec):
    return {k: torch.sqrt(torch.sum(v ** 2, dim=-2, keepdim=True)) for k, v in D12_vec.items()}


class Activation(nn.Module):
    def __init__(self, mult, activation_fn, dtype=torch.float32):
        super().__init__()
        if activation_fn == "gelu":
            self.activation_fn = nn.functional.gelu
        elif activation_fn == "relu":
            self.activation_fn = nn.functional.relu
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=dtype), requires_grad=False) for k, v in D12_Q.items()
        })

    def forward(self, D12_vec):
        D12_vec_perm = {k: self.activation_fn(self.Q[k].T @ v) for k, v in D12_vec.items()}
        return {k: self.Q[k] @ v for k, v in D12_vec_perm.items()}


class LayerNorm_invariant(nn.Module):
    def __init__(self, mult):
        super().__init__()
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False) for k, v in D12_Q.items()
        })
        self.gamma = nn.ParameterDict({
            k: nn.Parameter(torch.ones(v, dtype=torch.float32), requires_grad=True) for k, v in mult.items()
        })
        self.beta = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(v, dtype=torch.float32), requires_grad=True) for k, v in mult.items()
        })

    def normalize(self, v, eps=1e-5):
        # var, mean = torch.var_mean(v, dim=-1, keepdim=True)
        var, mean = torch.var_mean(v, dim=(-2,-1), keepdim=True)
        return (v - mean) / torch.sqrt(var + eps)

    def forward(self, D12_vec):
        return {k: self.Q[k] @ (self.normalize(self.Q[k].T @ v) * self.gamma[k] + self.beta[k]) for k, v in D12_vec.items()}


class D12_FC_out(nn.Module):
    def __init__(self, mult, D12_Q, dtype=torch.float32):
        super().__init__()
        self.linear3 = D12_linear(mult, {k: 1 for k in mult})
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=dtype), requires_grad=False) for k, v in D12_Q.items()
        })
        self.keys = ["A1", "B2", "E1", "E2", "E3", "E4", "E5"]
    
    def forward(self, D12_vec):
        D12_vec = self.linear3(D12_vec)
        D12_perm = torch.concat([self.Q[k].T @ D12_vec[k] for k in self.keys], dim=-1)
        return torch.mean(D12_perm, dim=-1)


class EmbedHead(nn.Module):
    def __init__(
        self,
        input_mult,
        inner_mult_1,
        inner_mult_2,
        out_mult,
        D12_Q
    ):
        super().__init__()
        self.layers = nn.Sequential(
            D12_featurize(D12_Q),
            D12_linear(input_mult, inner_mult_1),
            Activation(inner_mult_1, "gelu"),
            D12_linear(inner_mult_1, inner_mult_2),
            Activation(inner_mult_2, "gelu"),
            D12_linear(inner_mult_2, out_mult)
        )

    def forward(self, x):
        return self.layers(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, mult_model, num_heads):
        super().__init__()
        for v in mult_model.values():
            assert v % num_heads == 0, "multiplicity must be divisible by num_heads"
        
        self.mult_model = mult_model
        self.num_heads = num_heads
        self.mult_k = {k: v // num_heads for k, v in mult_model.items()}
        self.keys = ["A1", "B2", "E1", "E2", "E3", "E4", "E5"]
        self.W_q = D12_linear(self.mult_model, self.mult_model)
        self.W_k = D12_linear(self.mult_model, self.mult_model)
        self.W_v = D12_linear(self.mult_model, self.mult_model)
        self.W_o = D12_linear(self.mult_model, self.mult_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        Q = torch.cat([Q[k] for k in self.keys], dim=-1) # batch_size, self.num_heads, seq_length, sum(dim_rep * self.mult_k)
        K = torch.cat([K[k] for k in self.keys], dim=-1) 
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = {k: torch.matmul(attn_probs, v) for k, v in V.items()}
        return output
         
    def split_heads(self, D12_vec):
        heads = dict()
        for k, v in D12_vec.items():
            batch_size, seq_length, dim_rep, mult = v.size()
            heads[k] = (v
                .reshape(batch_size, seq_length, dim_rep, self.num_heads, self.mult_k[k])
                .permute(0, 3, 1, 2, 4) # batch_size, self.num_heads, seq_length, dim_rep, self.mult_k
                .reshape(batch_size, self.num_heads, seq_length, -1) # batch_size, self.num_heads, seq_length, dim_rep * self.mult_k
            )
        return heads
        
    def combine_heads(self, heads):
        D12_vec = dict()
        for k, v in heads.items():
            batch_size, _, seq_length, self.mult_k_rep = v.size()
            D12_vec[k] = (v
                .reshape(batch_size, self.num_heads, seq_length, -1, self.mult_k[k])
                # batch_size, self.num_heads, seq_length, dim_rep, self.mult_k
                .permute(0, 2, 3, 1, 4) # batch_size, seq_length, dim_rep, self.num_heads, self.mult_k
                .reshape(batch_size, seq_length, -1, self.mult_model[k])
            )
        return D12_vec
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, mult_model, mult_ff):
        super().__init__()
        self.fc1 = D12_linear(mult_model, mult_ff)
        self.fc2 = D12_linear(mult_ff, mult_model)
        self.relu = Activation(mult_ff, "gelu")

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, mult_model, num_heads, mult_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(mult_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(mult_model, mult_ff)
        self.norm1 = lambda x: x
        self.norm2 = lambda x: x
        # self.norm1 = LayerNorm_invariant(mult_model)
        # self.norm2 = LayerNorm_invariant(mult_model)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1({k: x[k] + attn_output[k] for k in x})
        ff_output = self.feed_forward(x)
        x = self.norm2({k: x[k] + ff_output[k] for k in x})
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, mult_model, num_heads, mult_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(mult_model, num_heads)
        self.cross_attn = MultiHeadAttention(mult_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(mult_model, mult_ff)
        # self.norm1 = LayerNorm_invariant(mult_model)
        # self.norm2 = LayerNorm_invariant(mult_model)
        # self.norm3 = LayerNorm_invariant(mult_model)
        self.norm1 = lambda x: x
        self.norm2 = lambda x: x
        self.norm3 = lambda x: x
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1({k: x[k] + self.dropout(attn_output[k]) for k in x})
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2({k: x[k] + self.dropout(attn_output[k]) for k in x})
        ff_output = self.feed_forward(x)
        x = self.norm3({k: x[k] + self.dropout(ff_output[k]) for k in x})
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, mult_model):
        super(PositionalEncoding, self).__init__()
        self.mult_model = mult_model
        self.ones = nn.Parameter(torch.ones(12,1), requires_grad=False)
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False) for k, v in D12_Q.items()
        })

    def forward(self, x):
        encoded_x = dict()
        for k, v in x.items():
            max_len = v.size(1)
            pe = torch.zeros(max_len, 1, self.mult_model[k])
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.mult_model[k], 2).float() * -(math.log(10000.0) / self.mult_model[k]))
            pe[:, :, 0::2] = torch.sin(position * div_term).unsqueeze(1)
            pe[:, :, 1::2] = torch.cos(position * div_term).unsqueeze(1)
            pe = pe.unsqueeze(0).to(v.device)
            encoded_x[k] = v + self.Q[k] @ (self.ones @ pe)
        return encoded_x
    

class VQVAE(nn.Module):
    def __init__(self, vocab_mult, model_mult, num_heads, num_layers, mult_ff, dropout, codebook_size):
        super().__init__()
        d_codebook = 12
        self.codebook_size = codebook_size
        self.encoder_embedding = EmbedHead(vocab_mult, model_mult, model_mult, model_mult, D12_Q)
        self.positional_encoding = PositionalEncoding(model_mult)
        self.encoder_layers = nn.ModuleList([EncoderLayer(model_mult, num_heads, mult_ff, dropout) for _ in range(num_layers)])
        self.encoder_output = D12_FC_out(model_mult, D12_Q)
        self.codebook = nn.Embedding(codebook_size, d_codebook)
        self.codebook.weight.data.uniform_(-1/d_codebook, 1/d_codebook)
        self.decoder_embedding = EmbedHead(vocab_mult, model_mult, model_mult, model_mult, D12_Q)
        self.decoder_layers = nn.ModuleList([EncoderLayer(model_mult, num_heads, mult_ff, dropout) for _ in range(num_layers)])
        self.decoder_output = D12_FC_out(model_mult, D12_Q)

    def encode(self, x):
        embedding = self.positional_encoding(self.encoder_embedding(x))
        for i, enc_layer in enumerate(self.encoder_layers):
            embedding = enc_layer(embedding, None)
        return self.encoder_output(embedding)
    
    def vq_indices(self, z):
        distance = (z.unsqueeze(2) - self.codebook.weight.unsqueeze(0).unsqueeze(0)).pow(2).mean(dim=-1)
        _, indices = torch.min(distance, dim=-1)
        return indices
    
    def vq_one_hot(self, z):
        indices = self.vq_indices(z)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=self.codebook_size).float()
        return one_hot
    
    def vq(self, z):
        return self.codebook(self.vq_indices(z))

    def decode(self, z):
        embedding = self.positional_encoding(self.decoder_embedding(z))
        for i, dec_layer in enumerate(self.decoder_layers):
            embedding = dec_layer(embedding, None)
        return torch.sigmoid(self.decoder_output(embedding))
    
    def forward(self, x):
        # x: [batch_size, seq_length, vocab_size]
        z = self.encode(x)
        z_vq = self.vq(z)
        z_straight_through = (z_vq - z).detach() + z
        x_recon = self.decode(z_straight_through)
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x)
        embed_loss = nn.functional.mse_loss(z_vq, z.detach())
        commit_loss = nn.functional.mse_loss(z, z_vq.detach())
        return x_recon, recon_loss, embed_loss, commit_loss

class Transformer(nn.Module):
    def __init__(self, mult_input, mult_model, num_heads, num_layers, mult_ff, dropout, D12_Q, d_codebook):
        super(Transformer, self).__init__()
        self.encoder_embedding = EmbedHead(mult_input, mult_model, mult_model, mult_model, D12_Q)
        self.decoder_embedding = EmbedHead(mult_input, mult_model, mult_model, mult_model, D12_Q)
        self.positional_encoding = PositionalEncoding(mult_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(mult_model, num_heads, mult_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(mult_model, num_heads, mult_ff, dropout) for _ in range(num_layers)])

        self.fc = D12_FC_out(mult_model, D12_Q)
        self.fc_vq_out = nn.Linear(12, d_codebook)
        self.fc_vq_in = nn.Linear(d_codebook, 12)

    def generate_mask(self, src, tgt):
        src_mask = (torch.sum(src, dim=2) > 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (torch.sum(tgt, dim=2) > 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        
        d = 1
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=d)).bool().to(src.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.positional_encoding(self.encoder_embedding(src))
        tgt_embedded = self.positional_encoding(self.decoder_embedding(self.fc_vq_in(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        output = self.fc(dec_output)
        logit = self.fc_vq_out(output)
        return logit


def train_main_loop(transformer, vqvae, optim, trainset, validset, lr, n_epoch, device, patience):
    wait = 0
    min_valid_loss = float('inf')
    for ep in tqdm(range(n_epoch)):
        transformer.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lr*(1-ep/n_epoch)
        # train
        criterion = nn.CrossEntropyLoss()
        for idx, src, tgt in trainset:
            optim.zero_grad()
            tgt = tgt.to(device)
            src = src.to(device)
            src_mask, tgt_mask = transformer.generate_mask(src, tgt[:, :-1, :])
            tgt_indices = vqvae.vq_indices(vqvae.encode(tgt))
            tgt_one_hot = torch.nn.functional.one_hot(tgt_indices, num_classes=vqvae.codebook_size).float()
            output = transformer(src, tgt_one_hot[:, :-1, :])
            loss = criterion(output.contiguous().view(-1, vqvae.codebook_size), tgt_indices[:, 1:].contiguous().view(-1))
            loss_train = loss.item()
            loss.backward()
            optim.step()
            
        # validation
        transformer.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, src, tgt in validset:
                tgt = tgt.to(device)
                src = src.to(device)
                src_mask, tgt_mask = transformer.generate_mask(src, tgt[:, :-1, :])
                tgt_indices = vqvae.vq_indices(vqvae.encode(tgt))
                tgt_one_hot = torch.nn.functional.one_hot(tgt_indices, num_classes=vqvae.codebook_size).float()
                output = transformer(src, tgt_one_hot[:, :-1, :], src_mask, tgt_mask)
                loss = criterion(output.contiguous().view(-1, vqvae.codebook_size), tgt_indices[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
        avg_valid_loss = total_loss / len(validset)

        # early stopping
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            torch.save(transformer.state_dict(), f"model_best_autoreg.pt")
            print(f'epoch {ep}, train_loss: {loss_train:.4f}, valid loss: {avg_valid_loss:.4f}')
            wait = 0
        else:
            print(f'epoch {ep}, train_loss: {loss_train:.4f}, valid loss: {avg_valid_loss:.4f}, min_valid_loss: {min_valid_loss:.4f}, wait: {wait} / {patience}')
            wait += 1
        if wait >= patience:
            break

# hardcoding these here
n_epoch = 1000
n_T = 1000
n_feat = 128
lr = 1e-4
ws_test = [0.0, 0.5, 2.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

src_vocab_size = 12
tgt_vocab_size = 128
d_model = 512
num_heads = 8
num_layers = 4
d_ff = 4096//8
max_seq_length = 2400
dropout = 0.1
batchsize = 16
mode = "train"


if exists("trainset_w.pkl") and exists("validset_w.pkl") and exists("testset_w.pkl"):
    print("splitted dataset found!")
    with open("trainset_w.pkl", "rb") as f:
        trainset = pickle.load(f)
    with open("validset_w.pkl", "rb") as f:
        validset = pickle.load(f)
    with open("testset_w.pkl", "rb") as f:
        testset = pickle.load(f)
else:
    print("?")

def collate_fn(batch):
    # Unpack batch into individual components
    idx, src_data, tgt_data, w = zip(*batch)
    #print(len(rates[0]), len(tgt_data[0]), len(src_data[0]))
    
    # Convert `src_data`, `tgt_data`, and `rates` to tensors if they are not already
    src_data = [torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s for s in src_data]
    tgt_data = [torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t for t in tgt_data]

    tgt_data = [torch.cat([torch.zeros(1, 12), t], dim=0) for t in tgt_data]

    # Pad src_data
    src_data = nn.utils.rnn.pad_sequence(src_data, batch_first=True, padding_value=0.).to(DEVICE)

    # Pad tgt_data
    tgt_data = nn.utils.rnn.pad_sequence(tgt_data, batch_first=True, padding_value=0).to(DEVICE)

    # Extract the last dimension and one-hot encode it
    return idx, src_data, tgt_data


trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

lr = 1e-4
input_mult = {k: 1 for k in D12_Q}
model_mult = {k: 64 for k in D12_Q}
vqvae = VQVAE(input_mult, model_mult, num_heads, 1, model_mult, dropout, 144).to(DEVICE)
optim = torch.optim.Adam(vqvae.parameters(), lr=lr)
vqvae.load_state_dict(torch.load("model_best_vqvae_equi.pt"))
transformer = Transformer(input_mult, model_mult, num_heads, num_layers, model_mult, dropout, D12_Q, 144).to(DEVICE)
train_main_loop(transformer, vqvae, optim, trainset, validset, lr, n_epoch, DEVICE, 20)
# eval_vqvae(vqvae, "model_best_vqvae_equi.pt", testset)