from transformer103_vqvae3 import VQVAE2
from module102 import *
from tqdm import tqdm
import torch
import pickle
from os.path import exists
from torch.utils import data
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
        src_mask = (torch.sum(src, dim=2) > 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (torch.sum(tgt, dim=2) > 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        
        d = 1
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=d)).bool().to(src.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for i, dec_layer in enumerate(self.decoder_layers):
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

def collate_fn(batch):
    # Unpack batch into individual components
    idx, src_data, tgt_data, w = zip(*batch)
    
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
            src_mask, tgt_mask = transformer.generate_mask(src, tgt[:, :-1, :])
            src_embedding = vqvae.get_src_embedding(src)
            tgt_indices = vqvae.vq_indices(vqvae.encode(src_embedding, tgt))
            #print(tgt_indices.min(), tgt_indices.max())
            tgt_one_hot = torch.nn.functional.one_hot(tgt_indices, num_classes=vqvae.codebook_size).float()
            output = transformer(src, tgt_one_hot[:, :-1, :], src_mask, tgt_mask)
            # print(tgt_indices, output)
            loss = criterion(output.contiguous().view(-1, vqvae.codebook_size), tgt_indices[:, 1:].contiguous().view(-1))
            loss_train = loss.item()
            #print(loss_train)
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
                src_embedding = vqvae.get_src_embedding(src)
                tgt_indices = vqvae.vq_indices(vqvae.encode(src_embedding, tgt))
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

def eval_main_loop(transformer, vqvae, checkpoint, testset, rate=0.5):
    transformer.load_state_dict(torch.load(checkpoint))
    transformer.eval()
    x_gens = []
    count = 0
    with torch.no_grad():
        for idx, src, tgt in tqdm(testset, total=len(testset)):
            # if count > 10:
            #     break
            src_embedding = vqvae.get_src_embedding(src)
            tgt_enc = vqvae.vq_one_hot(vqvae.encode(src_embedding,tgt))
            sampled_indices = []
            current_tgt_enc = tgt_enc[:, :1, :]
            for t in range(1, tgt_enc.size(1)):
                output = transformer(src, current_tgt_enc, None, None).detach()
                tgt_enc_prediction = torch.softmax(output[:, -1, :], dim=-1)
                # sample from categorical distribution
                sampled_index = torch.multinomial(tgt_enc_prediction, 1)
                sampled_indices.append(sampled_index)
                sampled_one_hot = torch.nn.functional.one_hot(sampled_index, num_classes=vqvae.codebook_size).float()
                current_tgt_enc = torch.cat([current_tgt_enc, sampled_one_hot], dim=1)
            x_gen = vqvae.decode(src_embedding, vqvae.codebook(torch.cat(sampled_indices, dim=1)))
            x_gen = (x_gen >= rate).long()
            x_gens.append((idx, x_gen))
            count += 1

    torch.save(x_gens, "song_test_music103.pt")

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


n_epoch = 200
src_vocab_size = 12
tgt_vocab_size = 12
d_model = 512
num_heads = 8
num_layers = 4
d_ff = 4096//8
max_seq_length = 2400
dropout = 0.1
batchsize = 16
lr = 1e-4

trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)


vqvae = VQVAE2(tgt_vocab_size, 256, num_heads, 2, d_ff, dropout, 4, 4).to(DEVICE)
vqvae.load_state_dict(torch.load("model_best_vqvae.pt"))
transformer = Transformer(src_vocab_size, vqvae.codebook_size, d_model, num_heads, 3, d_ff, max_seq_length, dropout).to(DEVICE)
optim = torch.optim.Adam(transformer.parameters(), lr=lr)
train_main_loop(transformer, vqvae, optim, trainset, validset, lr, n_epoch, DEVICE, 20)
eval_main_loop(transformer, vqvae, "model_best_autoreg.pt", testset)