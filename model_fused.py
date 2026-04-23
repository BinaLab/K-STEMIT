"""Single-branch fused GNN baselines used alongside the multi-branch K-STEMIT models."""

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM, EvolveGCNH, EvolveGCNO, GConvLSTM, TGCN, A3TGCN, DCRNN
from GAT_LSTM import GConvLSTM_GAT
from SAGE_LSTM import GConvSAGE_LSTM
from torch_geometric_temporal.nn.attention import TemporalConv
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm



class IceSheetModel_GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE
        if self.ADAPTIVE:
            self.evolve = EvolveGCNH(args.NODE_COUNT * args.BATCH_SIZE, args.FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        #self.evolve = EvolveGCNO(FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        self.conv = GConvLSTM(args.FEATURE_COUNT, args.DIMENSIONALITIES[0], 1, normalization="sym", bias=True)

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)



    def forward(self, graph_collection):

        h, c = None, None

        for i in range(self.LAYER_FEATURE_COUNT):            
            if self.ADAPTIVE:
                e = self.evolve(graph_collection[i].x.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
            else:
                e = graph_collection[i].x.float()

            h, c = self.conv(e, graph_collection[0].edge_index, graph_collection[0].edge_weight.float(), H=h, C=c)

        h = F.hardswish(h)

        h = self.lin1(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin2(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin3(h)
        return h
    


class IceSheetModel_GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        if self.ADAPTIVE:
            self.evolve = EvolveGCNH(args.NODE_COUNT * args.BATCH_SIZE, args.FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        #self.evolve = EvolveGCNO(FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        self.conv = GConvLSTM_GAT(args.FEATURE_COUNT, args.DIMENSIONALITIES[0], 1, normalization="sym", bias=True)

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)



    def forward(self, graph_collection):

        h, c = None, None

        for i in range(self.LAYER_FEATURE_COUNT):            
            if self.ADAPTIVE:
                e = self.evolve(graph_collection[i].x.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
            else:
                e = graph_collection[i].x.float()

            h, c = self.conv(e, graph_collection[0].edge_index, graph_collection[0].edge_weight.float(), H=h, C=c)

        h = F.hardswish(h)

        h = self.lin1(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin2(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin3(h)
        return h
    

class IceSheetModel_SAGE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE
        if self.ADAPTIVE:
            self.evolve = EvolveGCNH(args.NODE_COUNT * args.BATCH_SIZE, args.FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        #self.evolve = EvolveGCNO(FEATURE_COUNT, improved=True, cached=False, normalize=False, add_self_loops=True)
        self.conv = GConvSAGE_LSTM(args.FEATURE_COUNT, args.DIMENSIONALITIES[0], 1, normalization="sym", bias=True)
        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)



    def forward(self, graph_collection):

        h, c = None, None

        for i in range(self.LAYER_FEATURE_COUNT):            
            if self.ADAPTIVE:
                e = self.evolve(graph_collection[i].x.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
            else:
                e = graph_collection[i].x.float()

            h, c = self.conv(e, graph_collection[0].edge_index, graph_collection[0].edge_weight.float(), H=h, C=c)

        h = F.hardswish(h)

        h = self.lin1(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin2(h)
        h = F.hardswish(h)

        h = F.dropout(h, p=0.2, training=self.training)

        h = self.lin3(h)
        return h
