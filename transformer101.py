import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pandas as pd
from tqdm import tqdm
from os.path import exists
from os import remove, chdir
import pickle
from torch.utils.tensorboard import SummaryWriter
# from synthesizer import parse_csv, synthesize # functions in synthesizer.ipynb, PrettyMiDI and MuseScore needed

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
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src):
        src_mask = (torch.sum(src, dim=2) > 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        output = self.fc(enc_output)
        return torch.sigmoid(output)

if __name__ == "__main__":
    src_vocab_size = 12
    tgt_vocab_size = 12
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 4096//8
    max_seq_length = 2400
    dropout = 0.1
    batchsize = 8
    mode = "train"

    writer = SummaryWriter('.log-tf2')

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(DEVICE)

    n = 0
    for p in transformer.parameters():
        n += p.numel()
    print(n)
    # load data
    if exists("trainset_w.pkl") and exists("validset_w.pkl") and exists("testset_w.pkl"):
        print("splitted dataset found!")
        with open("trainset_w.pkl", "rb") as f:
            trainset = pickle.load(f)
        with open("validset_w.pkl", "rb") as f:
            validset = pickle.load(f)
        with open("testset_w.pkl", "rb") as f:
            testset = pickle.load(f)
    else:
        print("splitted dataset (unzip) not found!")
        dataset = []
        for i in tqdm(range(1, 910)):
            idx = f"{i:0>3}"
            
            data_frame = pd.read_csv(f'POP909/POP909/{idx}/melody_chord_1_beat.csv')

            data_frame['melody'] = data_frame['melody'].apply(lambda x: [float(n.strip().rstrip('.')) for n in x.strip('[]').split(',') if n.strip()])
            data_frame['chord'] = data_frame['chord'].apply(lambda x: [float(n.strip().rstrip('.')) for n in x.strip('[]').split(' ') if n.strip()])

            # If you want to extract these columns as lists of lists
            melody_list = data_frame['melody'].tolist()
            chord_list = data_frame['chord'].tolist()
            old_chord = None
            weight_list = []
            for chord in chord_list:
                if chord != old_chord:
                    weight_list.append(1.)
                else:
                    weight_list.append(0.)
                old_chord = chord

            # # Define a default value for NaNs
            # default_value = 2

            src_data = torch.tensor(melody_list)
            # Add an additional dimension at the front
            tgt_data = torch.tensor(chord_list)
            # Add an additional dimension at the front
            weights = torch.tensor(weight_list)
            dataset.append((idx, src_data, tgt_data, weights))

        generator = torch.Generator().manual_seed(0)
        trainset, validset, testset = data.random_split(dataset, [709, 100, 100], generator)
        with open("trainset_w.pkl", "wb") as f:
            pickle.dump(trainset, f)
        with open("validset_w.pkl", "wb") as f:
            pickle.dump(validset, f)
        with open("testset_w.pkl", "wb") as f:
            pickle.dump(testset, f)

    def collate_fn(batch):
        idx, src_data, tgt_data, weights = zip(*batch)
        src_data = nn.utils.rnn.pad_sequence(src_data, batch_first=True, padding_value=0.).to(DEVICE)
        tgt_data = nn.utils.rnn.pad_sequence(tgt_data, batch_first=True, padding_value=0.).to(DEVICE)
        weights = nn.utils.rnn.pad_sequence(weights, batch_first=True, padding_value=0.).to(DEVICE)
        return idx, src_data, tgt_data, weights
    
    
    trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
    validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
    testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    def loss_fn(output, target, weights):
        flat_weight = 1 + weights.repeat_interleave(12, 1).flatten()
        bce_loss = nn.BCELoss(weight=flat_weight, reduction="mean")(output.contiguous().view(-1), target.contiguous().view(-1))
        return bce_loss
    
    if mode == "train": 
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        
        val_cnt = 0
        beat_epoch = 0
        best_val_loss = float("inf")
        for epoch in range(2000):
            transformer.train()
            for i, pair_data in enumerate(trainset):
                _, src_data, tgt_data, weights = pair_data
                optimizer.zero_grad()
                output = transformer(src_data)
                loss = loss_fn(output, tgt_data, weights)
                loss.backward()
                optimizer.step()
            
            transformer.eval()
            
            total_val_loss = 0
            for i, pair_data in enumerate(validset):
                _, src_data, tgt_data, weights = pair_data
                output = transformer(src_data).detach()
                val_loss = loss_fn(output, tgt_data, weights).detach().item()
                total_val_loss += val_loss
            total_val_loss /= 100
            writer.add_scalar("train_loss", loss.item(), global_step=epoch)
            writer.add_scalar("valid_loss", total_val_loss, global_step=epoch)
            
            if total_val_loss < best_val_loss:
                if epoch > 0:
                    remove("model2_best.pt")
                best_val_loss = total_val_loss
                best_epoch = epoch
                val_cnt = 0
                print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Valid Loss: {total_val_loss} (best)")
                torch.save(transformer, f"model2_best.pt")
            else:
                val_cnt += 1
                print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Valid Loss: {total_val_loss} ({val_cnt}/50)")
                if val_cnt >= 50:
                    break

            if (epoch + 1) % 50 == 0:
                torch.save(transformer, f"model2_{epoch + 1}.pt")

    elif mode == "eval":
        transformer.eval()
        transformer.load_state_dict(torch.load("model2_best.pt").state_dict())
        # tot_loss = 0
        for i, pair_data in tqdm(enumerate(testset)):
            idx, src_data, tgt_data, weights = pair_data
            idx = idx[0]
            output = transformer(src_data, tgt_data).detach()
            
            print(loss_fn(output, tgt_data, weights))
            chords = torch.round(output).squeeze().cpu().numpy().tolist()
            real_chords, beats, durations = parse_csv(f"POP909/POP909/{idx}/melody_chord_1_beat.csv")
            synthesize(f"POP909/POP909/{idx}/{idx}.mid", chords, beats, durations, f"../gen/test_m2/test_{idx}_m2mlp_best_ns.mid", to_mp3=False)

