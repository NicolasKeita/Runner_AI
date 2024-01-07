import numpy as np
from torch import nn
import torch


class Network(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, layers: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)

        # List to add all layers to.
        model = list()

        in_dims = [in_dims] + layers
        out_dims = layers + [out_dims]

        for idx, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # Possibly use LazyLinear but not an issue.
            model.append(nn.Linear(in_dim, out_dim))

            # Check if its not he last layer, if it isn't add the activation
            # Maybe add a param to set this function, but not worth the to-do
            if idx < len(layers):
                model.append(nn.ReLU())  # TODO: Test ['nn.LeakyReLU()']

        # Combine into a model
        self.main = nn.Sequential(*model)

        # Check if running on a system that can compile, I can't test if this will actually work or what it may break.
        try:
            self.main = torch.compile(self.main)
        except RuntimeError:
            print("Compiling is not supported on this platform")

    def forward(self, values):
        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=torch.float)
        return self.main(values)


if __name__ == "__main__":
    network = Network(10, 2, [64, 128, 32])
    print(network)
