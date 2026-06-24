# DeepSEA Recreation

A from-scratch PyTorch reimplementation of [**DeepSEA**](https://www.nature.com/articles/nmeth.3547) (Zhou & Troyanskaya, *Nature Methods*, 2015) — a convolutional neural network that predicts chromatin features directly from raw DNA sequence.

This is a scaled-down educational recreation: instead of DeepSEA's 919 chromatin features across the whole genome, this version predicts **5 foundational epigenetic marks** from a handful of human chromosomes. The goal was to understand the model end-to-end — the biology, the data pipeline, the architecture, and *why* a downscaled version behaves the way it does.

A detailed write-up of the process, experiments, and what I learned is in [**REFLECTION.pdf**](./REFLECTION.pdf).

---

## What the model predicts

The network takes a **1,000 bp** one-hot-encoded DNA window and predicts the presence of 5 chromatin marks in its central region:

| Mark | What it signals |
|------|-----------------|
| **CTCF** | Insulator protein binding; directs chromatin loop extrusion. Highly sequence-specific. |
| **H3K4me3** | Tri-methylation marking active **promoters** near transcription start sites. |
| **H3K27ac** | Acetylation marking active **enhancers and promoters**. |
| **H3K4me1** | Mono-methylation marking **primed enhancers**. |
| **DNase** | Physical chromatin **accessibility** ("open" regulatory regions). |

Combinations are biologically meaningful — e.g. *H3K4me1 + H3K27ac + DNase* indicates an active enhancer, while *H3K4me3 + H3K27ac + DNase* indicates an active promoter.

---

## Architecture

A three-block 1D CNN followed by two fully connected layers (`deepsea.py`):

```
Input  (B, 4, 1000)              one-hot DNA: A/C/G/T channels
 ├─ Conv1d(4→320, k=8)  → ReLU → MaxPool(4) → Dropout(0.2)
 ├─ Conv1d(320→480, k=8) → ReLU → MaxPool(4) → Dropout(0.2)
 ├─ Conv1d(480→960, k=8) → ReLU → Dropout(0.5)
 ├─ Flatten  (B, 960×53)
 ├─ Linear(50880 → 925) → ReLU
 └─ Linear(925 → 5)               5 logits (sigmoid at inference)
```

Each convolutional filter acts like a learned **Position Weight Matrix**, scanning the sequence for motifs; max pooling provides translation invariance so motifs are detected regardless of exact position. Training uses **binary cross-entropy** (multi-label) via `BCEWithLogitsLoss`.

---

## Data pipeline

Input data (`data.py`):

- **Genome:** human reference FASTA files (chr18, chr19, chr21, chr22).
- **Labels:** [ENCODE](https://www.encodeproject.org/) peak files for the **GM12878** cell line (`.broadPeak` / `.narrowPeak`), one per mark.

Processing steps:

1. Load each chromosome's sequence; one-hot encode (`A=<1,0,0,0>`, etc.; unknown `N` → `<0,0,0,0>`).
2. Slide a 1,000 bp window in 200 bp steps. A window is **positive** for a mark if that mark's peak overlaps the central 200 bp.
3. Discard windows with >100 unknown (`N`) bases.
4. Balance the set to a **1:1 positive:negative** ratio.
5. **Chromosome-held-out split** — train on chr19/21/22, test on chr18 — to prevent leakage between nearby windows.

Output tensors: `X` of shape `(num_samples, 4, 1000)`, `Y` of shape `(num_samples, 5)`.

---

## Repository layout

```
deepsea.py        Model definition (DeepSEA CNN)
data.py           Builds train/test tensors from FASTA + ENCODE peaks
train.py          Training loop (Adam + ReduceLROnPlateau, AUROC reporting)
train.ipynb       Notebook version of training
train2.ipynb      Later experiment notebook
REFLECTION.pdf    Full write-up of the process, experiments, and findings
```

> **Note:** Large artifacts are intentionally **not** committed (see `.gitignore`): the genome FASTA files, ENCODE peak files, generated `*.pt` tensors, and model checkpoints. Several exceed GitHub's 100MB file limit. Regenerate them by following the steps below.

---

## Reproducing

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pandas scikit-learn

# 2. Download data into ./data/
#    - Reference FASTA: chr18, chr19, chr21, chr22 (UCSC hg19)
#    - ENCODE GM12878 peak files for H3K4me1, H3K4me3, H3K27ac, DNase, CTCF
#    (see data.py for the exact expected filenames)

# 3. Build the training tensors (writes X_train.pt, Y_train.pt, X_test.pt, Y_test.pt)
python data.py

# 4. Train (saves best_model.pt by validation loss)
python train.py
```

---

## Results & findings

Across three iterations, the model never pushed mean AUROC much above chance — and the **reason was the most interesting part of the project**:

- **Attempt 1** (chr21 only, positional 80/20 split): mean AUROC ≈ 0.60, with DNase highest (~0.67). Early validation plateau suggested overfitting.
- **Attempt 2** (added chr22, `ReduceLROnPlateau`): textbook overfitting — training loss fell while validation loss rose; AUROC ≈ 0.48. The positional split was **leaking** adjacent windows across train/val.
- **Attempt 3** (proper chromosome-held-out split, added negatives): a more honest test, AUROC plateaued ≈ 0.52.

The core limitation is structural, not a bug: with only **5 biologically correlated marks**, positive windows almost always activate 2–3 marks together, collapsing the within-window discriminative signal that DeepSEA's 919-feature setting relies on. Combined with training on just three chromosomes, the model has little room to generalize.

**Highest-impact next steps:** train on the full genome (minus held-out test chromosomes), add L2 weight decay and stronger dropout, and expand to a broader set of chromatin features to restore discriminative signal.

The full reasoning is in [REFLECTION.pdf](./REFLECTION.pdf).

---

## References

- Zhou, J. & Troyanskaya, O. *Predicting effects of noncoding variants with deep learning–based sequence model.* Nature Methods 12, 931–934 (2015).
- [ENCODE Project](https://www.encodeproject.org/) — GM12878 chromatin data.
