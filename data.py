import pandas as pd
import torch
import numpy as np

MARKS = ['H3k4me1', 'H3k4me3', 'H3k27ac', 'DNase', 'CTCF']

PEAK_FILES = [
    "data/wgEncodeBroadHistoneGm12878H3k4me1StdPk.broadPeak",
    "data/wgEncodeBroadHistoneGm12878H3k4me3StdPk.broadPeak",
    "data/wgEncodeBroadHistoneGm12878H3k27acStdPk.broadPeak",
    "data/wgEncodeUwDnaseGm12878PkRep1.narrowPeak",
    "data/wgEncodeUwTfbsGm12878CtcfStdPkRep1.narrowPeak",
]

TRAIN_CHROMOSOMES = [
    ('chr19', 'data/chr19.fa'),
    ('chr21', 'data/chr21.fa'),
    ('chr22', 'data/chr22.fa'),
]

TEST_CHROMOSOMES = [
    ('chr18', 'data/chr18.fa'),
]


def load_fasta(path):
    lines = open(path, 'r').read().splitlines()
    return ''.join(l for l in lines if not l.startswith('>')).upper()


def one_hot(seq):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    return np.array([mapping.get(base, [0,0,0,0]) for base in seq], dtype=np.float32)


peak_dfs = [pd.read_csv(p, sep="\t", header=None) for p in PEAK_FILES]
print(f"Loaded {len(peak_dfs)} ENCODE peak files")


def _coords_for_chrom(chrom):
    out = []
    for df in peak_dfs:
        sub = df[df[0] == chrom]
        out.append((np.array(sub[1]), np.array(sub[2])))
    return out


def build_windows(chromosomes, neg_ratio=1.0):
    """
    neg_ratio: number of negative windows to keep per positive window.
    Set to 1.0 for 1:1 positive:negative balance (DeepSEA default).
    """
    X_pos, Y_pos, X_neg, Y_neg = [], [], [], []
    for chrom, fasta_path in chromosomes:
        seq = load_fasta(fasta_path)
        print(f"{chrom}: {len(seq):,} bp")
        coords = _coords_for_chrom(chrom)
        n = 500
        while n < len(seq) - 500:
            label = np.array(
                [np.any((s < n + 100) & (e > n - 100)) for s, e in coords],
                dtype=np.float32,
            )
            window = seq[n - 500:n + 500]
            if window.count('N') <= 100:
                if label.any():
                    X_pos.append(one_hot(window))
                    Y_pos.append(label)
                else:
                    X_neg.append(one_hot(window))
                    Y_neg.append(label)
            n += 200
        print(f"  pos so far: {len(X_pos):,} | neg so far: {len(X_neg):,}")

    n_neg = int(len(X_pos) * neg_ratio)
    rng = np.random.default_rng(42)
    neg_idx = rng.choice(len(X_neg), size=min(n_neg, len(X_neg)), replace=False)
    X_neg = [X_neg[i] for i in neg_idx]
    Y_neg = [Y_neg[i] for i in neg_idx]

    X_all = X_pos + X_neg
    Y_all = Y_pos + Y_neg
    print(f"  final: {len(X_pos):,} pos + {len(X_neg):,} neg = {len(X_all):,} total")
    return np.stack(X_all), np.stack(Y_all)


print("=== Building train set (chr19, chr21, chr22) ===")
X_train, Y_train = build_windows(TRAIN_CHROMOSOMES)
print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)

print("\n=== Building test set (chr18) ===")
X_test, Y_test = build_windows(TEST_CHROMOSOMES)
print("X_test shape:", X_test.shape, "Y_test shape:", Y_test.shape)

print("\npositives per mark (train):")
for m, c in zip(MARKS, Y_train.sum(axis=0)):
    print(f"  {m:8s} {int(c):>7d}  ({c/len(Y_train):.1%})")

print("\npositives per mark (test):")
for m, c in zip(MARKS, Y_test.sum(axis=0)):
    print(f"  {m:8s} {int(c):>7d}  ({c/len(Y_test):.1%})")

torch.save(X_train, 'X_train.pt')
torch.save(Y_train, 'Y_train.pt')
torch.save(X_test,  'X_test.pt')
torch.save(Y_test,  'Y_test.pt')
torch.save(MARKS, 'marks.pt')
