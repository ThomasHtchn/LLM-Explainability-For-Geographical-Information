import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import time


class ActivationDataset(Dataset):
    def __init__(self, df):
        self.data = df["mean_pooling"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        x = torch.tensor(x, dtype=torch.float32).view(-1)

        return x


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity=1e-3):
        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sparsity = sparsity

    def forward(self, x):
        z = self.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def loss_fn(x, x_hat, z, sparsity_weight):
    recon_loss = ((x - x_hat) ** 2).mean()
    sparsity_loss = z.abs().mean()
    return recon_loss + sparsity_weight * sparsity_loss

def normalize_loss_fn(x, x_hat, z, sparsity_weight):
    recon_loss = (((x_hat - x) ** 2).mean(dim=1) / (x ** 2).mean(dim=1)).mean()
    sparsity_loss = (z.abs().sum(dim=1) / (x.norm(dim=1))).mean() # + 1e-8
    return recon_loss + sparsity_weight * sparsity_loss


def train(model, loader, optimizer, device, sparsity_weight):
    model.train()

    total_loss = 0

    for x in tqdm(loader):

        x = x.to(device)

        x_hat, z = model(x)

        #loss = loss_fn(x, x_hat, z, sparsity_weight)
        #loss = topk_loss_fn(x, x_hat) # topk
        loss = normalize_loss_fn(x, x_hat, z, sparsity_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #model.normalize_decoder() # tpopk

        total_loss += loss.item()

    return total_loss / len(loader)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="sae.pt")
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--earlystop_patience", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=2048)

    args = parser.parse_args()


    df = pd.read_pickle(args.input_pkl)

    print("Loaded data")

    dataset = ActivationDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    sample = dataset[0]
    input_dim = sample.shape[0]

    hidden_dim = args.hidden_dim * input_dim

    print(f"input_dim = {input_dim}")
    print(f"hidden_dim = {hidden_dim}")

    name_prefix = (args.input_pkl).split("act")[1][:-4]
    sparsity_str = str(args.sparsity).split(".")[1]
    output_path = f"models/sae{name_prefix}_dim{hidden_dim}_sf{sparsity_str}"

    hidden_dim = args.hidden_dim * input_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, sparsity=args.sparsity).to(device)
    #model = SparseAutoencoderTopK(input_dim=input_dim, hidden_dim=hidden_dim, k=args.top_k).to(device) # topk

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=args.scheduler_patience, min_lr=1e-6)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6) # topk

    # early stopping
    best_loss = float("inf")
    early_stopping_patience = args.earlystop_patience
    epochs_without_improvement = 0

    start_time = time.time()
    for epoch in range(args.epochs):

        loss = train(model, loader, optimizer, device, args.sparsity)

        scheduler.step(loss)
        #scheduler.step() # topk
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Epoch {epoch} | loss: {loss:.6f} | lr: {current_lr:.2e}")

        if loss < best_loss:
            best_loss = loss
            epochs_without_improvement = 0
    
        else:
            epochs_without_improvement += 1
        
        # early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after "
                f"{epoch + 1} epochs"
            )
            break
    training_time = time.time() - start_time 

    output_path = output_path + f"_epochs{epoch + 1}.pt"
    torch.save(model.state_dict(), output_path)

    print("Saved model:", output_path)
    print(f"Training time: {int(training_time // 60)} m {int(training_time % 60)} s")


if __name__ == "__main__":
    main()