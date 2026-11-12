"""Two-layer PyG R-GCN with typed input projections and a song-country decoder."""

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class RevivalRGCN(nn.Module):
    def __init__(self, input_dims, pair_dim, hidden=32, dropout=0.1, prevalence=0.003):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, hidden) for dim in input_dims])
        self.layers = nn.ModuleList(
            [
                RGCNConv(hidden, hidden, num_relations=6, aggr="mean", is_sorted=True)
                for _ in range(2)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 2 + pair_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.decoder[-1].bias, float(torch.logit(torch.tensor(prevalence))))
        nn.init.normal_(self.decoder[-1].weight, std=0.01)

    def forward(self, batch):
        features = [batch["song_x"], batch["artist_x"], batch["country_x"]]
        x = torch.cat(
            [projection(values) for projection, values in zip(self.projections, features)], dim=0
        )
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x, batch["edge_index"], batch["edge_type"])))
        country_offset = len(features[0]) + len(features[1])
        pair = torch.cat(
            [x[batch["song"]], x[batch["country"] + country_offset], batch["pair_x"]], dim=1
        )
        return self.decoder(pair).squeeze(1)
