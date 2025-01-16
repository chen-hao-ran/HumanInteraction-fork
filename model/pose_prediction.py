import torch
import torch.nn as nn

class pose_prediction(nn.Module):
    def __init__(self, smpl, frame_length=16, input_dim=72, output_dim=72, hidden_dim=256, num_layers=3, dropout=0.2):
        super(pose_prediction, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.smpl = smpl

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        pred = self.model(x)

        return pred
