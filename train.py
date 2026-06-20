from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from deepsea import DeepSEA
import torch.nn as nn
import torch
import math
import os

DATA_DIR = '/kaggle/input/datasets/madhavmandala/deepsea-data' if os.path.exists('/kaggle') else '.'

def load(fname):
    return torch.from_numpy(torch.load(os.path.join(DATA_DIR, fname), weights_only=False)).float()

X_train = load('X_train.pt').permute(0, 2, 1)
Y_train = load('Y_train.pt')
X_test  = load('X_test.pt').permute(0, 2, 1)
Y_test  = load('Y_test.pt')

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_test,  Y_test),  batch_size=512, shuffle=False)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = DeepSEA(n_outputs=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
)
criterion = nn.BCEWithLogitsLoss()

marks = ['H3k4me1', 'H3k4me3', 'H3k27ac', 'DNase', 'CTCF']

print(f"Device: {device}")
print(f"Train samples: {len(X_train):,} (chr19+21+22) | Test samples: {len(X_test):,} (chr18)")
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
print("-" * 80)

best_val_loss = math.inf
log_interval  = max(1, len(train_loader) // 5)

for epoch in range(30):
    model.train()
    train_loss   = 0.0
    batches_seen = 0
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss   += loss.item() * len(xb)
        batches_seen += 1

        if (batch_idx + 1) % log_interval == 0:
            running_loss = train_loss / (batches_seen * train_loader.batch_size)
            pct = 100.0 * (batch_idx + 1) / len(train_loader)
            print(f"  Epoch {epoch+1:>2d} [{pct:5.1f}%] batch {batch_idx+1}/{len(train_loader)} "
                  f"| running loss {running_loss:.4f}")

    train_loss /= len(X_train)

    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            all_logits.append(model(xb.to(device)).cpu())
            all_labels.append(yb)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    val_loss = criterion(logits, labels).item()
    probs    = torch.sigmoid(logits).numpy()

    aurocs = []
    for i in range(5):
        try:
            aurocs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            aurocs.append(float('nan'))

    mean_auroc = sum(a for a in aurocs if not math.isnan(a)) / sum(1 for a in aurocs if not math.isnan(a))

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1:>2d} | train {train_loss:.4f} | val {val_loss:.4f} | "
          f"mean AUROC {mean_auroc:.3f} | lr {current_lr:.2e} | "
          + " ".join(f"{m}={a:.3f}" for m, a in zip(marks, aurocs)))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch':      epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_loss':   val_loss,
            'aurocs':     dict(zip(marks, aurocs)),
        }, '/kaggle/working/best_model.pt')
        print(f"  --> Saved checkpoint (val loss improved to {val_loss:.4f})")

print("-" * 80)
print(f"Training complete. Best val loss: {best_val_loss:.4f} | checkpoint: best_model.pt")
