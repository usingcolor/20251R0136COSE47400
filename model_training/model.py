import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3):
        super(MLPModel, self).__init__()

        if num_layers == 3:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, output_dim),
            )
        elif num_layers == 5:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim),
            )
        elif num_layers == 7:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 2048),
                nn.BatchNorm1d(2048),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim),
            )

    def forward(self, x):
        return self.layers(x)
