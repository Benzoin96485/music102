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
#from pretty_midi import PrettyMIDI, instrument_name_to_program, Instrument, note_name_to_number, Note
from os import system
# from synthesizer import parse_csv, synthesize # functions in synthesizer.ipynb, PrettyMiDI and MuseScore needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-9
VERSION="3-7"

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

class D12_linear(nn.Module):
    def __init__(self, input_mult, output_mult):
        super().__init__()
        self.din = sum(input_mult.values())
        self.dout = sum(output_mult.values())
        self.linear = nn.Linear(self.din, self.dout)
        self.Q = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False) for k, v in D12_Q.items()
        })
        self.keys = ["A1", "B2", "E1", "E2", "E3", "E4", "E5"]
        self.splits = [output_mult[k] for k in self.keys]

    def forward(self, D12_vec):
        out = dict()
        D12_perm = torch.concat([self.Q[k].T @ D12_vec[k] for k in self.keys], dim=-1)
        D12_perm = self.linear(D12_perm)
        D12_perm_split = torch.split(D12_perm, self.splits, dim=-1)
        for i, k in enumerate(self.keys):
            out[k] = self.Q[k] @ D12_perm_split[i]
        return out

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
            k: nn.Parameter(torch.ones(v, dtype=torch.float32), requires_grad=False) for k, v in mult.items()
        })
        self.beta = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(v, dtype=torch.float32), requires_grad=False) for k, v in mult.items()
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
        self.norm1 = LayerNorm_invariant(mult_model)
        self.norm2 = LayerNorm_invariant(mult_model)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1({k: x[k] + attn_output[k] for k in x})
        ff_output = self.feed_forward(x)
        x = self.norm2({k: x[k] + ff_output[k] for k in x})
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
    

class Transformer(nn.Module):
    def __init__(self, mult_input, mult_model, num_heads, num_layers, mult_ff, dropout, D12_Q):
        super(Transformer, self).__init__()
        self.encoder_embedding = EmbedHead(mult_input, mult_model, mult_model, mult_model, D12_Q)
        self.positional_encoding = PositionalEncoding(mult_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(mult_model, num_heads, mult_ff, dropout) for _ in range(num_layers)])

        self.fc = D12_FC_out(mult_model, D12_Q)

    def generate_mask(self, src):
        src_mask = (torch.sum(src, dim=2) > 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.positional_encoding(self.encoder_embedding(src))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        output = self.fc(enc_output)
        logit = torch.sigmoid(output)
        return logit


def parse_csv(csv_file):
    df = pd.read_csv(csv_file)
    chords = df["chord"].apply(lambda x: eval(x.replace(".", ","))).to_list()
    beats = df["time"].to_list()
    durations = df["duration"].to_list()
    return chords, beats, durations


def synthesize(midi_file, chords, beats, durations, output, instrument="String Ensemble 1", velocity=50, group=3, melody_only=True, to_mp3=False):
    KEYS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
    midi_data = PrettyMIDI(midi_file)
    if melody_only:
        midi_data.instruments = [midi_data.instruments[0]]
    program = instrument_name_to_program(instrument)
    accompany = Instrument(program=program, name="accompany")
    old_chord = []
    for chord, beat, duration in zip(chords + [[]], beats + [0], durations + [0]):
        if chord != old_chord:
            if old_chord:
                for bit, key in zip(old_chord, KEYS):
                    if bit:
                        note_number = note_name_to_number(f"{key}{group}")
                        note = Note(velocity=velocity, pitch=note_number, start=old_beat, end=old_beat+total_duration)
                        accompany.notes.append(note)
            old_beat = beat
            total_duration = duration
            old_chord = chord
        else:
            total_duration += duration
    midi_data.instruments.append(accompany)
    midi_data.write(output)
    if to_mp3:
        system(f"Musescore4 {output} -o {output.replace('mid', 'mp3')}")


if __name__ == "__main__":
    input_mult = {k: 1 for k in D12_Q}
    model_mult = {k: 64 for k in D12_Q}
    num_heads = 8
    num_layers = 4
    dropout = 0
    batchsize = 8
    mode = "train"

    writer = SummaryWriter(f'.log-tf{VERSION}')

    transformer = Transformer(input_mult, model_mult, num_heads, num_layers, model_mult, dropout, D12_Q).to(DEVICE)

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
    
    
    trainset = data.DataLoader(trainset, batch_size=16, collate_fn=collate_fn)
    validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
    testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    def loss_fn(output, target, weights):
        flat_weight = 1 + weights.repeat_interleave(12, 1).flatten()
        bce_loss = nn.BCELoss(weight=flat_weight, reduction="mean")(output.contiguous().view(-1), target.contiguous().view(-1))
        return bce_loss
    
    if mode == "train": 
        optimizer = optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-10)
        val_cnt = 0
        beat_epoch = 0
        best_val_loss = float("inf")
        for epoch in range(2000):
            transformer.train()
            for i, pair_data in tqdm(enumerate(trainset)):
                _, src_data, tgt_data, weights = pair_data
                optimizer.zero_grad()
                output = transformer(src_data)
                loss = loss_fn(output, tgt_data, weights)
                loss.backward()
                writer.add_scalar("loss", loss, global_step=epoch + (i + 1) * batchsize / 710)
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
                    remove(f"model{VERSION}_best.pt")
                best_val_loss = total_val_loss
                best_epoch = epoch
                val_cnt = 0
                print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Valid Loss: {total_val_loss} (best)")
                torch.save(transformer, f"model{VERSION}_best.pt")
            else:
                val_cnt += 1
                print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Valid Loss: {total_val_loss} ({val_cnt}/50)")
                if val_cnt >= 50:
                    break

            if (epoch + 1) % 50 == 0:
                torch.save(transformer, f"model{VERSION}_{epoch + 1}.pt")

    elif mode == "eval":
        transformer.eval()
        transformer.load_state_dict(torch.load("model3-7_best.pt", map_location=DEVICE).state_dict())
        # tot_loss = 0
        for i, pair_data in tqdm(enumerate(testset)):
            idx, src_data, tgt_data, weights = pair_data
            idx = idx[0]
            if idx != "017":
                continue
            output = transformer(src_data).detach()
            
            print(loss_fn(output, tgt_data, weights))
            chords = torch.round(output).squeeze().cpu().numpy().tolist()
            real_chords, beats, durations = parse_csv(f"POP909/POP909/{idx}/melody_chord_1_beat.csv")
            synthesize(f"POP909/POP909/{idx}/{idx}.mid", chords, beats, durations, f"test_{idx}_music102.mid", to_mp3=False)
            