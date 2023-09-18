from torch import nn
from devices.set.models.mlp import MLP
from torch_geometric.nn import MessagePassing, GCNConv, GatedGraphConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F

class GAT_Model(nn.Module):
    def __init__(self, input_dim, fc_dims, heads,dropout_p=0.4):
        super(GAT_Model, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))
        layers=[]
        i = 0
        for dim in fc_dims:
            if i < len(fc_dims)-1:
                layers.append(GATConv(input_dim, dim, heads=heads[i],dropout=dropout_p))
            else:
                layers.append(GATConv(input_dim, dim, heads=heads[i], concat=False,dropout=dropout_p))

            input_dim = dim*heads[i]
            i +=1
        self.layers = layers
        self.length = len(heads)

    def forward(self, data, edge_index):
        
        for i in range(self.length):
            data = self.layers[i](data,edge_index)


        return data
