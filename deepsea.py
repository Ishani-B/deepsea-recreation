import torch
import torch.nn as nn

class DeepSEA(nn.Module):
    def __init__(self, n_outputs=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=8),       # (B, 320, 993)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # (B, 320, 248)  pooling -> invariance
            nn.Dropout(0.2),

            nn.Conv1d(320, 480, kernel_size=8),     # (B, 480, 241)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # (B, 480, 60)
            nn.Dropout(0.2),

            nn.Conv1d(480, 960, kernel_size=8),     # (B, 960, 53)
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        # 960 channels x 53 positions = 50880 for a 1000bp input
        self.fc = nn.Sequential(
            nn.Linear(960 * 53, 925),   # every neuron sees all conv outputs (full window)
            nn.ReLU(),                  # ReLU(Wx)
            nn.Linear(925, n_outputs),  # -> 5 logits
        )

    def forward(self, x):           # x: (B, 4, 1000)
        x = self.conv(x)
        x = x.flatten(1)            # (B, 50880)
        return self.fc(x)           # logits, (B, 5)