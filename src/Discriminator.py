from torch import nn


class MLPDiscriminator(nn.Module):
    def __init__(self, layer_sizes):
        super(MLPDiscriminator, self).__init__()

        layers = []
        for idx in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))

        self.mlp = nn.Sequential(*layers)

        self.project = nn.Linear(layer_sizes[-1], 1)

    def forward(self, x):
        # Flatten input
        output = x.view(x.size(0), -1)

        output = self.mlp(output)

        output = self.project(output)
        return output
