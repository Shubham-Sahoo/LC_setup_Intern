from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from policies.features import Features, EdgeFeatures

########################################################################################################################
# region GNN Utilities
########################################################################################################################
"""
Nodes are represented as numbers, eg. 0
Edges are represented as dashes and {}, eg. --{0}--

   0  --{0}--   1  --{1}--   2   --{2}--  ...  --{C-2}--  C-1
   |            |     ...    |            ...              |
{2C-1}         {2C}   ...  {2C+1}         ...            {3C-3}
   |            |     ...    |            ...              |
   C --{C-1}-- C+1 --{C}--  C+2  --{C+1}-- ... --{2C-2}-- 2C-1

number of nodes = N = 2C
number of edges = E = 3C-2
"""


def create_edge_index(C: int) -> np.ndarray:
    """
    Creates the edge index.

    Args:
        C (int): number of camera rays.
    Returns:
        edge_index (np.ndarray, dtype=np.int, shape=(2, E): the edge index for GNNs.
    """
    h1 = np.vstack([np.arange(0, C - 1),
                    np.arange(1, C)]).astype(np.int)
    h2 = np.vstack([np.arange(C, 2 * C - 1),
                    np.arange(C + 1, 2 * C)]).astype(np.int)
    v  = np.vstack([np.arange(0, C),
                    np.arange(C, 2 * C)]).astype(np.int)
    edge_index = np.hstack([h1, h2, v])  # (2, E)
    return edge_index


def create_node_and_edge_feats(feats: Features) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates node and edge features.

    Returns:
        nodes (np.ndarray, dtype=np.float32, shape=(B, N, 2): node features that includes intensity and drb.
        edges (np.ndarray, dtype=np.float32, shape=(B, E, 4): edge features that includes dx, dz, dr, dt.
    """
    # node features
    i   = np.atleast_3d(np.stack([feats.i1,  feats.i2],  axis=-1))  # (B, C, 2)
    drb = np.atleast_3d(np.stack([feats.drb, feats.drb], axis=-1))  # (B, C, 2)
    nodes = np.concatenate([i, drb], axis=1)  # (B, N, 2) where N = 2C

    # edge features
    def _create_edge_features(edge_feats: EdgeFeatures) -> np.ndarray:
        """
        Returns:
            edges (np.ndarray, dtype=np.float32, shape=(B, C/C-1, 4): edge features for edges corresponding
        """
        return np.atleast_3d(np.stack([np.atleast_2d(edge_feats.dx),
                             np.atleast_2d(edge_feats.dz),
                             np.atleast_2d(edge_feats.dr),
                             np.atleast_2d(edge_feats.dt)], axis=-1))  # (B, C/C-1, 4)

    h_edges_1 = _create_edge_features(feats.eh1)
    h_edges_2 = _create_edge_features(feats.eh2)
    v_edges   = _create_edge_features(feats.ev)
    edges = np.concatenate([h_edges_1, h_edges_2, v_edges], axis=1)  # (B, E, 4) where E = 3C-2

    return nodes, edges

# endregion
########################################################################################################################
# region Graph Neural Networks
########################################################################################################################


class EdgeConv(MessagePassing):
    def __init__(self,
                 C: int,
                 in_channels: int,
                 out_channels: int):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(nn.Linear(2 * in_channels + 4, out_channels),
                                 nn.ReLU(),
                                 nn.Linear(out_channels, out_channels))

        self.edge_index = torch.from_numpy(create_edge_index(C))  # (2, E)

    def forward(self,
                x: torch.Tensor,
                e: torch.Tensor) -> td.Distribution:
        """
        Args:
            x (torch.Tensor, dtype=torch.float32, shape=(B, N, in_channels): input node features
            e (torch.Tensor, dtype=torch.float32, shape=(B, E, 4): input edge features
        Returns:
            out (torch.Tensor, dtype=torch.float32, shape=(B, N, out_channels): output of the graph convolution.
        """
        return self.propagate(self.edge_index, x=x, e=e)  # (B, N, out_channels)

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                e: torch.Tensor):
        """
        Args:
            x_i (torch.Tensor, dtype=torch.float32, shape=(B, E, in_channels): features of central nodes
            x_j (torch.Tensor, dtype=torch.float32, shape=(B, E, in_channels): features of neighboring nodes
            e (torch.Tensor, dtype=torch.float32, shape=(B, E, 4): edge features.

        Returns:
            msg (torch.Tensor, dtype=torch.float32, shape=(B, E, out_channels): message.
        """
        mlp_input = torch.cat([x_i, x_j, e], dim=2)  # (B, E, 2*in_channels + 4)
        return self.mlp(mlp_input)  # (B, E, out_channels)


@register_network
class GNN(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.C = C

        self.conv1 = EdgeConv(C, 2, 4)
        self.conv2 = EdgeConv(C, 4, 8)
        self.conv3 = EdgeConv(C, 8, 16)

        self.mu = nn.Linear(32, 1)
        self.sigma = nn.Linear(32, 1)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"{self.__class__.__name__}: has {num_params} parameters")

    def forward(self,
                feats: Features):
        x, e = create_node_and_edge_feats(feats)  # (B, N, 2) and (B, E, 4)
        x = torch.from_numpy(x)  # (B, N, 2)
        e = torch.from_numpy(e)  # (B, E, 4)

        x = self.conv1(x, e)  # (B, N, 4)
        x = self.conv2(x, e)  # (B, N, 8)
        x = self.conv3(x, e)  # (B, N, 8)

        x = torch.cat([x[:, :self.C, :],
                       x[:, self.C:, :]], dim=2)  # (B, C, 16)

        mu = self.mu(x).squeeze(-1)  # (B, C)
        # sigma = 1e-5 + torch.relu(self.sigma(x)).squeeze(-1)  # (B, C)
        sigma = 1

        pi = td.Normal(loc=mu, scale=sigma)  # batch_shape=[B, C] event_shape=[]
        pi = td.Independent(pi, 1)  # batch_shape=[B,] event_shape=[C,]
        return pi

# endregion
########################################################################################################################
