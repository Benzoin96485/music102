from module102 import *
from tqdm import tqdm
from os.path import exists
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data


class VQVAE2(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, codebook_size, d_codebook):
        super().__init__()
        self.codebook_size = codebook_size
        self.d_codebook = d_codebook
        self.embedding_src = EmbedHead(vocab_size, d_model, d_model, d_model)
        self.encoder_embedding_tgt = EmbedHead(vocab_size, d_model, d_model, d_model)
        self.decoder_embedding_tgt = EmbedHead(d_codebook, d_model, d_model, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        self.layers_src = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_layers_tgt = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_output = nn.Linear(d_model, d_codebook)

        self.codebook = nn.Embedding(codebook_size, d_codebook)
        self.codebook.weight.data.uniform_(-1/d_codebook, 1/d_codebook)
        
        self.decoder_layers_tgt = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)


    def get_src_embedding(self, src):
        src_embedding = self.dropout(self.positional_encoding(self.embedding_src(src)))
        for i, enc_layer in enumerate(self.layers_src):
            src_embedding = enc_layer(src_embedding, None)
        return src_embedding

    def encode(self, src_embedding, tgt):
        tgt_embedding = self.dropout(self.positional_encoding(self.encoder_embedding_tgt(tgt)))
        for i, dec_layer in enumerate(self.encoder_layers_tgt):
            tgt_embedding = dec_layer(tgt_embedding, src_embedding, None, None)
        return self.encoder_output(tgt_embedding)

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

    def decode(self, src_embedding, tgt):
        tgt_embedding = self.dropout(self.positional_encoding(self.decoder_embedding_tgt(tgt)))
        for i, dec_layer in enumerate(self.decoder_layers_tgt):
            tgt_embedding = dec_layer(tgt_embedding, src_embedding, None, None)
        return torch.sigmoid(self.decoder_output(tgt_embedding))
    
    def forward(self, src, tgt):
        # x: [batch_size, seq_length, vocab_size]
        src_embedding = self.get_src_embedding(src)
        z = self.encode(src_embedding, tgt)
        z_vq = self.vq(z)
        z_straight_through = (z_vq - z).detach() + z
        tgt_recon = self.decode(src_embedding, z_straight_through)
        recon_loss = nn.functional.binary_cross_entropy(tgt_recon, tgt)
        embed_loss = nn.functional.mse_loss(z_vq, z.detach())
        commit_loss = nn.functional.mse_loss(z, z_vq.detach())
        return tgt_recon, recon_loss, embed_loss, commit_loss


def train_VQVAE(vqvae, optim, trainset, validset, lr, n_epoch, device, patience, model_path, alpha=0.5, beta=1):
    wait = 0
    min_valid_loss = float('inf')
    for ep in tqdm(range(n_epoch)):
        vqvae.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lr*(1-ep/n_epoch)
        loss_ema = None
        # train
        for idx, src, tgt in trainset:
            optim.zero_grad()
            _, recon_loss, embed_loss, commit_loss = vqvae(tgt)
            loss = recon_loss + beta * embed_loss + alpha * commit_loss
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            optim.step()
            
        # validation
        vqvae.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, src, tgt in validset:
                tgt = tgt.to(device)
                src = src.to(device)
                _, recon_loss, embed_loss, commit_loss = vqvae(tgt)
                loss = recon_loss
                total_loss += loss.item()
        avg_valid_loss = total_loss / len(validset)

        # early stopping
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            torch.save(vqvae.state_dict(), model_path)
            print(f'epoch {ep}, train_loss: {loss_ema:.4f}, valid loss: {avg_valid_loss:.4f}')
            wait = 0
        else:
            print(f'epoch {ep}, train_loss: {loss_ema:.4f}, valid loss: {avg_valid_loss:.4f}, min_valid_loss: {min_valid_loss:.4f}, wait: {wait} / {patience}')
            wait += 1
        if wait >= patience:
            break

def collate_fn(batch):
    # Unpack batch into individual components
    idx, src_data, tgt_data, w = zip(*batch)
    
    # Convert `src_data`, `tgt_data`, and `rates` to tensors if they are not already
    src_data = [torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s for s in src_data]
    tgt_data = [torch.tensor(t, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t for t in tgt_data]

    # Pad src_data
    src_data = nn.utils.rnn.pad_sequence(src_data, batch_first=True, padding_value=0.).to(DEVICE)

    # Pad tgt_data
    tgt_data = nn.utils.rnn.pad_sequence(tgt_data, batch_first=True, padding_value=0).to(DEVICE)

    # Extract the last dimension and one-hot encode it
    return idx, src_data, tgt_data


def train_VQVAE(vqvae, optim, trainset, validset, lr, n_epoch, device, patience, model_path, alpha=0.5, beta=1):
    wait = 0
    min_valid_loss = float('inf')
    for ep in tqdm(range(n_epoch)):
        vqvae.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lr*(1-ep/n_epoch)
        loss_ema = None
        # train
        for idx, src, tgt in trainset:
            optim.zero_grad()
            _, recon_loss, embed_loss, commit_loss = vqvae(src, tgt)
            loss = recon_loss + beta * embed_loss + alpha * commit_loss
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            optim.step()
            
        # validation
        vqvae.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, src, tgt in validset:
                _, recon_loss, embed_loss, commit_loss = vqvae(src, tgt)
                loss = recon_loss
                total_loss += loss.item()
        avg_valid_loss = total_loss / len(validset)

        # early stopping
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            torch.save(vqvae.state_dict(), model_path)
            print(f'epoch {ep}, train_loss: {loss_ema:.4f}, valid loss: {avg_valid_loss:.4f}')
            wait = 0
        else:
            print(f'epoch {ep}, train_loss: {loss_ema:.4f}, valid loss: {avg_valid_loss:.4f}, min_valid_loss: {min_valid_loss:.4f}, wait: {wait} / {patience}')
            wait += 1
        if wait >= patience:
            break


def eval_VQVAE(vqvae, checkpoint, testset):
    vqvae.load_state_dict(torch.load(checkpoint))
    vqvae.eval()
    x_gens = []
    count = 0
    with torch.no_grad():
        for idx, src, tgt in tqdm(testset, total=len(testset)):
            if count > 10:
                break
            x_gen, _, _, _ = vqvae(src, tgt)
            x_gen = (x_gen >= 0.5).long()
            
            x_gens.append((idx, x_gen))
            count += 1

    torch.save(x_gens, "song_test_music103.pt")


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


    trainset = data.DataLoader(trainset, batch_size=batchsize, collate_fn=collate_fn)
    validset = data.DataLoader(validset, batch_size=1, collate_fn=collate_fn)
    testset = data.DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    lr = 1e-3
    vqvae = VQVAE2(tgt_vocab_size, 256, num_heads, 1, d_ff, dropout, 128, 12).to(DEVICE)
    optim = torch.optim.Adam(vqvae.parameters(), lr=lr)
    train_VQVAE(vqvae, optim, trainset, validset, lr, n_epoch, DEVICE, 20, "model_best_vqvae_128.pt")
    eval_VQVAE(vqvae, "model_best_vqvae_128.pt", testset)