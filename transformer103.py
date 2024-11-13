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
from synthesizer import parse_csv, synthesize

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
        self.decoder_embedding = EmbedHead(tgt_vocab_size, d_model, d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
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
    # def generate_mask(self, src, tgt):
    #     src_mask = (torch.sum(src, dim=2) >= 0).unsqueeze(1).unsqueeze(3)
    #     tgt_mask = (torch.sum(tgt, dim=2) >= 0).unsqueeze(1).unsqueeze(3)
    #     seq_length = tgt.size(1)
    #     src_length = src.size(1)
    #     d = torch.randint(-1, 7, (1,)).item()
    #     d2 = torch.randint(4, 16, (1,)).item()
    #     nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=d)).bool().to(DEVICE)
    #     local_mask = ((torch.triu(torch.ones(1, src_length, src_length), diagonal=-d2)).bool() & (1-torch.triu(torch.ones(1, src_length, src_length), diagonal=d2)).bool()).to(DEVICE)
    #     tgt_mask = tgt_mask & nopeak_mask
    #     src_mask = src_mask & local_mask
    #     return src_mask, tgt_mask

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
            # dec_output = dec_layer(dec_output, enc_output, src_mask[:, :, :-1,:], tgt_mask)

        output = self.fc(dec_output)
        return torch.sigmoid(output)

if __name__ == "__main__":
    chdir("POP909/model_eval")
    src_vocab_size = 12
    tgt_vocab_size = 12
    d_model = 512
    num_heads = 8
    num_layers = 4
    d_ff = 4096//8
    max_seq_length = 2400
    dropout = 0.1
    batchsize = 16
    mode = "eval"

    writer = SummaryWriter('.log')

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(DEVICE)

    # load data
    if exists("trainset_unzip.pkl") and exists("validset_unzip.pkl") and exists("testset_unzip.pkl"):
        print("splitted dataset found!")
        with open("trainset_unzip.pkl", "rb") as f:
            trainset = pickle.load(f)
        with open("validset_unzip.pkl", "rb") as f:
            validset = pickle.load(f)
        with open("testset_unzip.pkl", "rb") as f:
            testset = pickle.load(f)
    else:
        print("splitted dataset (unzip) not found!")
        dataset = []
        for i in tqdm(range(1, 910)):
            idx = f"{i:0>3}"
            
            data_frame = pd.read_csv(f'../POP909/{idx}/melody_chord_1_beat.csv')

            data_frame['melody'] = data_frame['melody'].apply(lambda x: [float(n.strip().rstrip('.')) for n in x.strip('[]').split(',') if n.strip()])
            data_frame['chord'] = data_frame['chord'].apply(lambda x: [float(n.strip().rstrip('.')) for n in x.strip('[]').split(' ') if n.strip()])

            # If you want to extract these columns as lists of lists
            melody_list = data_frame['melody'].tolist()
            chord_list = data_frame['chord'].tolist()

            # # Define a default value for NaNs
            # default_value = 2

            src_data = torch.tensor(melody_list)
            # Add an additional dimension at the front
            tgt_data = torch.tensor(chord_list)
            # Add an additional dimension at the front
            dataset.append((idx, src_data, tgt_data))

        generator = torch.Generator().manual_seed(0)
        trainset, validset, testset = data.random_split(dataset, [709, 100, 100], generator)
        with open("trainset_unzip.pkl", "wb") as f:
            pickle.dump(trainset, f)
        with open("validset_unzip.pkl", "wb") as f:
            pickle.dump(validset, f)
        with open("testset_unzip.pkl", "wb") as f:
            pickle.dump(testset, f)

    def collate_fn(batch):
        idx, src_data, tgt_data = zip(*batch)
        src_data = nn.utils.rnn.pad_sequence(src_data, batch_first=True, padding_value=0.).to(DEVICE)
        tgt_data = nn.utils.rnn.pad_sequence(tgt_data, batch_first=True, padding_value=0.).to(DEVICE)
        return idx, src_data, tgt_data
    
    
    trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
    validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
    testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    criterion = nn.BCELoss(reduction="mean")
    if mode == "train": 
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        
        transformer.train()

        for epoch in tqdm(range(200)):
            for i, pair_data in tqdm(enumerate(trainset)):
                idx, src_data, tgt_data = pair_data
                optimizer.zero_grad()
                output = transformer(src_data, tgt_data[:, :-1, :])
                loss = criterion(output.contiguous().view(-1), tgt_data[:, 1:, :].contiguous().view(-1))
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss", loss, global_step=epoch + (i + 1) * batchsize / 710)
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

            if (epoch + 1) % 10 == 0:
                torch.save(transformer, f"model_{epoch + 1}.pt")

    elif mode == "eval":
        transformer.eval()
        transformer.load_state_dict(torch.load("model2_best.pt").state_dict())
        # tot_loss = 0
        for i, pair_data in tqdm(enumerate(validset)):
            if i == 3:
                idx, src_data, tgt_data = pair_data
                idx = idx[0]
                output = transformer(src_data, tgt_data[:, :-1, :]).detach()
                
                output = torch.cat((tgt_data[:, 0:1, :], output), dim=1).detach()
                print(criterion(output.contiguous().view(-1), tgt_data.contiguous().view(-1)))
                chords = torch.round(output).squeeze().cpu().numpy().tolist()
                real_chords, beats, durations = parse_csv(f"../POP909/{idx}/melody_chord_1_beat.csv")
                synthesize(f"../POP909/{idx}/{idx}.mid", chords, beats, durations, f"test_{idx}_m2mlp_best_ns.mid", to_mp3=True)
        #         synthesize(f"{idx}/{idx}.mid", real_chords, beats, durations, f"test_{idx}_real.mid", to_mp3=True)
                

        #     tot_loss += criterion(output.contiguous().view(-1), tgt_data.contiguous().view(-1))
        # print(tot_loss / 100)
        for i, pair_data in tqdm(enumerate(validset)):
            if i == 3:
                idx, src_data, real_tgt_data = pair_data
                idx = idx[0]
                print(idx)
                tgt_data = torch.zeros(1, 1, 12, requires_grad=False).to(DEVICE)
                for _ in tqdm(range(src_data.shape[1] - 1)):
                    full_output = transformer(src_data, tgt_data)
                    output = full_output[:, -1:, :].detach()
                    tgt_data = torch.cat((tgt_data, output), dim=1).detach()
                print(criterion(tgt_data.contiguous().view(-1), real_tgt_data.contiguous().view(-1)))
                chords = torch.round(tgt_data).squeeze().cpu().numpy().tolist()
                real_chords, beats, durations = parse_csv(f"../POP909/{idx}/melody_chord_1_beat.csv")
                synthesize(f"../POP909/{idx}/{idx}.mid", chords, beats, durations, f"test_{idx}_m2mlp_best.mid", to_mp3=True)
                # synthesize(f"../POP909/{idx}/{idx}.mid", real_chords, beats, durations, f"test_{idx}_real.mid", to_mp3=True)
            
    # # Initialize target sequence with start token
    # start_token_id = 0  # Replace with your start token ID
    # end_token_id = 4097
    # src = src_data.to("cuda")
    # tgt = torch.tensor([[start_token_id]]).to("cuda")
    # # Set your maximum sequence length to avoid infinite loops
    # max_length = 600

    # for _ in tqdm(range(max_length)):
    #     output = transformer(src, tgt)
    #     next_token_id = torch.argmax(output, dim=-1)[:, -1]  # Take the last token from the sequence
    #     tgt = torch.cat((tgt, next_token_id.unsqueeze(-1)), dim=1)

    #     if next_token_id.item() == end_token_id:  # Replace with your end token ID
    #         break

    # print(tgt)