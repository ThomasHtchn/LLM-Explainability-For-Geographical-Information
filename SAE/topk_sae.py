import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class ActivationDataset(Dataset):
    def __init__(self, path, col="mean_pooling"):
        df = pd.read_pickle(path)

        xs = []
        for row in df[col].values:
            xs.append(torch.tensor(np.array(row), dtype=torch.float32))

        self.xs = torch.stack(xs)

        print("Dataset shape:", self.xs.shape)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        return self.xs[idx]


# ---------------------------------------------------------
# TopK SAE Model
# ---------------------------------------------------------
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()

        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)

        self.k = k

        # decoder normalization stability trick
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data, dim=0
            )

    def topk(self, x):
        vals, idx = torch.topk(x, self.k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, idx, F.relu(vals))
        return out

    def forward(self, x):
        z = self.encoder(x)
        z = self.topk(z)
        x_hat = self.decoder(z)
        return x_hat, z


# ---------------------------------------------------------
# Loss (normalized reconstruction)
# ---------------------------------------------------------
def recon_loss(x, x_hat):
    err = ((x - x_hat) ** 2).mean(dim=-1)
    norm = (x ** 2).mean(dim=-1) + 1e-8
    return (err / norm).mean()


# ---------------------------------------------------------
# Dead neuron tracker + resampling
# ---------------------------------------------------------
@torch.no_grad()
def resample_dead_neurons(model, dead_indices, batch_x):
    if len(dead_indices) == 0:
        return 0

    # for j in dead_indices:
    #     nn.init.kaiming_uniform_(
    #         model.encoder.weight.data[j:j+1],
    #         a=np.sqrt(5)
    #     )

    #     nn.init.kaiming_uniform_(
    #         model.decoder.weight.data[:, j:j+1].T,
    #         a=np.sqrt(5)
    #     )

    #     if model.encoder.bias is not None:
    #         model.encoder.bias.data[j] = 0
    for j in dead_indices:
        idx = torch.randint(
            0,
            batch_x.shape[0],
            (1,),
            device=batch_x.device
        )

        v = batch_x[idx].squeeze(0)
        v = F.normalize(v, dim=0)

        model.encoder.weight.data[j] = v
        model.encoder.bias.data[j] = 0

        model.decoder.weight.data[:, j] = v

    model.normalize_decoder()

    print(f"    => Resampled {len(dead_indices)} dead neurons")

    return len(dead_indices)


# ---------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------
def log_stats(z, total_steps, last_fired_step, resample_step):
    global_alive_fraction = (
        (total_steps - last_fired_step)
        <= resample_step
    ).float().mean().item()
    return {
        "mean_activation": z.abs().mean().item(),
        "max_activation": z.abs().max().item(),
        "active_fraction": (z > 0).float().mean().item(),
        "batch_alive_fraction": ((z > 0).any(dim=0)).float().mean().item(),
        "global_alive_fraction": global_alive_fraction
    }

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train(args):
    dataset = ActivationDataset(args.input_pkl)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    d_in = dataset.xs.shape[1]
    d_sae = d_in * args.hidden_dim
    k = args.top_k

    # Guard: k cannot exceed the SAE hidden dimension
    if k > d_sae:
        raise ValueError(
            f"--top_k ({k}) must be <= d_sae ({d_sae}). "
            f"Reduce --top_k or increase --hidden_dim."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"d_in={d_in} | d_sae={d_sae} | k={k}")

    model = TopKSAE(d_in, d_sae, k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=args.scheduler_patience)

    best_loss = float("inf")
    epochs_no_improve = 0
    total_steps = 0
    count_resampling = 0

    # For resampling dead neurons
    last_fired_step = torch.zeros(d_sae, dtype=torch.long, device=device)

    name_prefix = (args.input_pkl).split("act")[1][:-4]
    output_dir = args.output_dir
    output_path = f"{output_dir}topk_{args.top_k}_sae{name_prefix}_dim{d_sae}.pt" #_resample{args.resample_step}.pt"

    start_time = time.time()
    
    for epoch in range(args.epochs):
        # pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_loss = 0.0

        for step, x in enumerate(loader): #pbar):
            x = x.to(device)

            x_hat, z = model(x)
            loss = recon_loss(x, x_hat)

            opt.zero_grad()
            loss.backward()
            opt.step()

            model.normalize_decoder()

            total_loss += loss.item()

            stats = log_stats(z, total_steps, last_fired_step, args.resample_step)
            # pbar.set_description(
            #     f"Epoch {epoch} | loss {loss.item():.4f} | "
            #     #f"act {stats['mean_activation']:.4f} | "
            #     f"active {stats['active_fraction']:.3f}"
            #     # f"batch_alive_fraction {stats['batch_alive_fraction']:.3f}"
            #     # f"global_alive_fraction {stats['global_alive_fraction']:.3f}"
            # )

            # # Resampling dead neurons
            # if total_steps >= args.resample_step and total_steps % args.resample_step == 0:
            #     dead_mask = (total_steps - last_fired_step) >= args.resample_step
            #     dead_indices = torch.where(dead_mask)[0]
                
            #     n_dead = resample_dead_neurons(model, dead_indices, x)
            #     if n_dead > 0:
            #         count_resampling += 1
            #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=args.scheduler_patience)
            #         epochs_no_improve = 0
            #         best_loss = float("inf")
            #         print(f"       Reset scheduler and early stopping after resampling {n_dead} neurons")
            total_steps += 1

        epoch_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {epoch_loss:.6f} | loss {loss.item():.4f} | mean_act {stats['mean_activation']:.4f} | alive_frac {stats['active_fraction']:.3f} | batch_alive_fraction {stats['batch_alive_fraction']:.3f} | global_alive_fraction {stats['global_alive_fraction']:.3f}")

        scheduler.step(epoch_loss)

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            print(f"    New best loss {best_loss:.6f} - saved !")# - saved to {output_path}")
            torch.save(model.state_dict(), output_path)

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.earlystop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    training_time = time.time() - start_time 

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Training time: {int(training_time // 60)} m {int(training_time % 60)} s")
    print(f"Number of neurons resampling : {count_resampling}")
    

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a TopK Sparse Autoencoder on MLP activations.")
    parser.add_argument("--input_pkl",           type=str,   required=True,        help="Path to the pickled activation DataFrame.")
    parser.add_argument("--hidden_dim",          type=int,   default=8,            help="SAE width multiplier: d_sae = d_in * hidden_dim.")
    parser.add_argument("--batch_size",          type=int,   default=256,          help="Training batch size.")
    parser.add_argument("--epochs",              type=int,   default=500,          help="Maximum number of training epochs.")
    parser.add_argument("--lr",                  type=float, default=1e-3,         help="AdamW learning rate.")
    parser.add_argument("--scheduler_patience",  type=int,   default=10,           help="ReduceLROnPlateau patience (epochs).")
    parser.add_argument("--earlystop_patience",  type=int,   default=20,           help="Early-stopping patience (epochs).")
    parser.add_argument("--top_k",               type=int,   default=64,           help="Number of active features per forward pass (TopK k).")
    parser.add_argument("--resample_step",       type=int,   default=10000,        help="Number of steps before resampling dead neurons.")
    parser.add_argument("--output_dir",          type=str,   default="models/",     help="Dir path to save model at.")


    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()