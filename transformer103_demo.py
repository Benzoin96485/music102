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
#from torch.utils.tensorboard import SummaryWriter
#from synthesizer import parse_csv, synthesize

DEVICE = "cuda"

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        max_len = x.size(1)
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).to(x.device)
        return x + pe
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class EmbedHead(nn.Module):
    def __init__(
        self,
        input_dim,
        inner_dim_1,
        inner_dim_2,
        out_dim
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, inner_dim_1)
        self.linear2 = nn.Linear(inner_dim_1, inner_dim_2)
        self.linear3 = nn.Linear(inner_dim_2, out_dim)
        self.activation_fn = nn.functional.gelu

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        x = self.activation_fn(x)
        x = self.linear3(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = EmbedHead(src_vocab_size, d_model, d_model, d_model)
        self.decoder_embedding = EmbedHead(tgt_vocab_size + 1, d_model, d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size + 1)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (torch.sum(src, dim=2) >= 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (torch.sum(tgt, dim=2) >= 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        if self.training:
            d = torch.randint(-14, 2, (1,)).item()
        else:
            d = 1
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=d)).bool().to(DEVICE)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return torch.sigmoid(output)

if __name__ == "__main__":
    chdir("POP909/model_train")
    src_vocab_size = 12
    tgt_vocab_size = 12
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 4096//8
    max_seq_length = 2400
    dropout = 0.1
    batchsize = 16
    mode = "inference"

    #writer = SummaryWriter('.log')

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(DEVICE)

    # load data
    if exists("new_trainset.pkl") and exists("new_validset.pkl") and exists("new_testset.pkl"):
        print("splitted dataset found!")
        with open("new_trainset.pkl", "rb") as f:
            trainset = pickle.load(f)
        with open("new_validset.pkl", "rb") as f:
            validset = pickle.load(f)
        with open("new_testset.pkl", "rb") as f:
            testset = pickle.load(f)
    else:
        print("?")
    
    def collate_fn(batch):
        # Unpack batch into individual components
        idx, rates, tgt_data, src_data = zip(*batch)
        #print(len(rates[0]), len(tgt_data[0]), len(src_data[0]))
        
        # Convert `src_data`, `tgt_data`, and `rates` to tensors if they are not already
        src_data = [torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s for s in src_data]
        tgt_data = [torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t for t in tgt_data]
        rates = [torch.tensor(r, dtype=torch.float32) if not isinstance(r, torch.Tensor) else r for r in rates]
        
        #print(tgt_data[0].shape, rates[0].shape)
        # Concatenate `rates` as an additional feature to `src_data`
        src_data = [torch.cat([s], dim=-1) for s, r in zip(src_data, rates)]
        
        
        # Create `tgt_data_with_rates` with a one-step delay for `rates`
        tgt_data_with_rates = [torch.cat([t, r[:, None]], dim=-1) for t, r in zip(tgt_data, rates)]
        #print(tgt_data_with_rates[0].shape, src_data[0].shape)
        # Pad sequences to create uniform batch tensors
        src_data = nn.utils.rnn.pad_sequence(src_data, batch_first=True, padding_value=0.).to(DEVICE)
        tgt_data_with_rates = nn.utils.rnn.pad_sequence(tgt_data_with_rates, batch_first=True, padding_value=0.).to(DEVICE)
        #tgt_data = nn.utils.rnn.pad_sequence(tgt_data, batch_first=True, padding_value=0.).to(DEVICE)
        
        return idx, src_data, tgt_data_with_rates#, tgt_data

    
    trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
    validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
    testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    criterion = nn.BCELoss(reduction="mean")
    if mode == "train": 
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        transformer.train()

        for epoch in tqdm(range(200)):
            for i, pair_data in tqdm(enumerate(trainset)):
                idx, src_data, tgt_data_with_rates = pair_data
                optimizer.zero_grad()
                # Use tgt_data_with_rates as input to the transformer
                output = transformer(src_data, tgt_data_with_rates[:, :-1, :])
                #print(output.shape, tgt_data_with_rates.shape)
                loss = criterion(output.contiguous().view(-1), tgt_data_with_rates[:, 1:, :].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                '''
                # Match the shapes for loss calculation
                output = output.contiguous().view(-1)
                target = tgt_data_with_rates[:, 1:, :].contiguous().view(-1)

                # Ensure target and output have the same shape
                min_len = min(output.size(0), target.size(0))
                output = output[:min_len]
                target = target[:min_len]

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()'''

                #writer.add_scalar("loss", loss, global_step=epoch + (i + 1) * batchsize / 710)
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

            if (epoch + 1) % 10 == 0:
                torch.save(transformer, f"model_{epoch + 1}.pt")
    
    if mode == 'inference':
        # inference step
        transformer.eval()
        transformer.load_state_dict(torch.load("model_60.pt").state_dict())
        tot_loss = 0

        for i, pair_data in tqdm(enumerate(testset)):
            if i == 10:
                break
            idx, src_data, tgt_data_with_rates = pair_data
            idx = idx[0]
            #generated_tgt = []
            #print(tgt_data_with_rates.shape)
            current_tgt = tgt_data_with_rates[:, :1, :]  # Use the first step as a starting point
            timer = 1

            for t in range(1, tgt_data_with_rates.size(1)):  # Generate timestep by timestep
                output = transformer(src_data, current_tgt).detach()
                #print(output.shape)
                rate_prediction = output[:, -1, -1].item()  # Assuming the last feature is the rate
                tgt_prediction = output[:, -1, :]  # Assuming other features are `tgt_data`

                # Calculate the dynamic threshold for flipping
                flip_threshold = 1 / (timer + 1)

                # Determine where to flip
                #print(flip_threshold, rate_prediction)
                flip_mask = (rate_prediction < flip_threshold)

                # Update next target based on flip mask
                next_tgt = tgt_prediction.clone()
                probabilities = next_tgt[:, :12]  # First 12 values are probabilities
                sampled_binary = torch.bernoulli(probabilities)  # Randomly sample 0 or 1 based on probabilities
                next_tgt[:, :12] = sampled_binary
                #print(next_tgt)

                # Update the timer
                if flip_mask or np.random.random() < 0.2:
                    next_tgt[:, -1] = 1
                    timer = 1
                else:
                    next_tgt = current_tgt[:, -1, :].clone()
                    next_tgt[:, -1] = flip_threshold
                    timer += 1

                # generated_tgt.append(next_tgt)
                current_tgt = torch.cat([current_tgt, next_tgt.unsqueeze(1)], dim=1)


            # Compute loss if necessary
            loss = criterion(current_tgt[:, 1:, :].contiguous().view(-1), tgt_data_with_rates[:, 1:, :].contiguous().view(-1))
            tot_loss += loss.item()

            torch.save(current_tgt, f'../../song_{idx}.pt')
'''
            # Optional: Save or analyze the generated sequences
            print(f"Index: {idx}, Loss: {loss.item()}")
            

        print(f"Total Loss: {tot_loss / len(testset)}")
'''



