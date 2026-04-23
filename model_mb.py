"""Knowledge-informed multi-branch K-STEMIT models and ablation variants with physical features."""

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM, EvolveGCNH, EvolveGCNO, GConvLSTM, TGCN, A3TGCN, DCRNN
from GAT_LSTM import GConvLSTM_GAT
from SAGE_LSTM import GConvSAGE_LSTM
from torch_geometric_temporal.nn.attention import TemporalConv
from torch_geometric.nn import SAGEConv, GCNConv
    

class IceSheetModel_Multibranch1(torch.nn.Module):
    def __init__(self, args, clamp_fusion_weights=True):
        super().__init__()

        self.spectral_conv = GCNConv(42, args.DIMENSIONALITIES[0], improved=True)

        self.spatial_conv = SAGEConv(42, args.DIMENSIONALITIES[0])

        self.temporal_conv = TemporalConv(in_channels=8, out_channels=args.DIMENSIONALITIES[0], kernel_size=5)

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT 
        self.ADAPTIVE = args.ADAPTIVE
        self.clamp_fusion_weights = clamp_fusion_weights

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

        self.alpha = torch.nn.Parameter(torch.tensor(0.33))  # scalar between 0 and 1
        self.beta = torch.nn.Parameter(torch.tensor(0.33))  # scalar between 0 and 1



    def forward(self, graph_collection):

        h1, h2 = None, None
        static = []

        for i in range(self.LAYER_FEATURE_COUNT):   
            e = graph_collection[i].x[:, 2:].float()

            static.append(e)

        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
        h2 = self.spatial_conv(e.float(), graph_collection[0].edge_index)
        
        static = torch.stack(static, dim=0)
        static = static.unsqueeze(0)

        h3 = self.temporal_conv(static).squeeze()

        # h = torch.cat((h1, h2, h3), dim=1)
        if self.clamp_fusion_weights:
            self.alpha.data = torch.clamp(self.alpha.data, 0, 1)
            self.beta.data = torch.clamp(self.beta.data, 0, 1)

        h = h1 * self.alpha + h2 * self.beta + h3 * (1 - self.alpha - self.beta)

        h = F.hardswish(h)

        h = self.lin1(h)
        h = F.hardswish(h)

        h = self.lin2(h)
        h = F.hardswish(h)

        h = self.lin3(h)
        return h


class IceSheetModel_Multibranch1_NoClamp(IceSheetModel_Multibranch1):
    def __init__(self, args):
        super().__init__(args, clamp_fusion_weights=False)

class IceSheetModel_Multibranch_ablation2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GCNConv(42, args.DIMENSIONALITIES[0], improved=True)
        self.temporal_conv = TemporalConv(in_channels=8, out_channels=args.DIMENSIONALITIES[0], kernel_size=5)
        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT 

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))  # scalar between 0 and 1


    def forward(self, graph_collection):

        h1 = None
        static = []

        for i in range(self.LAYER_FEATURE_COUNT):   
            e = graph_collection[i].x[:, 2:].float()
            static.append(e)

        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())

        static = torch.stack(static, dim=0)
        static = static.unsqueeze(0)


        h3 = self.temporal_conv(static).squeeze()

        # h = torch.cat((h1, h3), dim=1)
        h = h1 * self.alpha + h3 * (1 - self.alpha)

        h = F.hardswish(h)

        h = self.lin1(h)
        h = F.hardswish(h)

        h = self.lin2(h)
        h = F.hardswish(h)

        h = self.lin3(h)
        return h
    

class IceSheetModel_Multibranch_ablation1_Old(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial_conv = SAGEConv(42, args.DIMENSIONALITIES[0])
        self.temporal_conv = TemporalConv(in_channels=8, out_channels=args.DIMENSIONALITIES[0], kernel_size=5)
        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0] * 2, args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)


    def forward(self, graph_collection):

        h2 = None
        static = []

        for i in range(self.LAYER_FEATURE_COUNT):   
            e = graph_collection[i].x[:, 2:].float()

            static.append(e)

        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        h2 = self.spatial_conv(e.float(), graph_collection[0].edge_index)
        
        static = torch.stack(static, dim=0)
        static = static.unsqueeze(0)
        h3 = self.temporal_conv(static).squeeze()
        
        h = torch.cat((h2, h3), dim=1)
        h = F.hardswish(h)
        h = self.lin1(h)
        h = F.hardswish(h)
        h = self.lin2(h)
        h = F.hardswish(h)
        h = self.lin3(h)
        return h
    

class IceSheetModel_Multibranch_ablation3(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GCNConv(42, args.DIMENSIONALITIES[0], improved=True)
        self.spatial_conv = SAGEConv(42, args.DIMENSIONALITIES[0])
    
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

        self.alpha = torch.nn.Parameter(torch.tensor(0.5))  # scalar between 0 and 1
        
    def forward(self, graph_collection):

        h1, h2 = None, None

        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
        h2 = self.spatial_conv(e.float(), graph_collection[0].edge_index)

        # h = torch.cat((h1, h2), dim=1)

        h = h1 * self.alpha + h2 * (1 - self.alpha)

        h = F.hardswish(h)
        h = self.lin1(h)
        h = F.hardswish(h)
        h = self.lin2(h)
        h = F.hardswish(h)
        h = self.lin3(h)
        return h
    

class IceSheetModel_Multibranch_ablation4(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GCNConv(42, args.DIMENSIONALITIES[0], bias=True, normalize='sym')
        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)


    def forward(self, graph_collection):

        h1 = None

        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, graph_collection[0].edge_weight.float())
        h1 = F.hardswish(h1)
        h1 = self.lin1(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin2(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin3(h1)
        return h1
    

class IceSheetModel_Multibranch_ablation5(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial_conv = SAGEConv(42, args.DIMENSIONALITIES[0])        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE
        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)
        
    def forward(self, graph_collection):
        h2 = None
        e = torch.cat( (graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        h2 = self.spatial_conv(e.float(), graph_collection[0].edge_index)
        h2 = F.hardswish(h2)
        h2 = self.lin1(h2)
        h2 = F.hardswish(h2)
        h2 = self.lin2(h2)
        h2 = F.hardswish(h2)
        h2 = self.lin3(h2)
        return h2
    

class IceSheetModel_Multibranch_ablation6(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temporal_conv = TemporalConv(in_channels=8, out_channels=args.DIMENSIONALITIES[0], kernel_size=5)
    
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT 
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

    def forward(self, graph_collection):

        static = []

        for i in range(self.LAYER_FEATURE_COUNT):   
            e = graph_collection[i].x[:, 2:].float()
            static.append(e)
        static = torch.stack(static, dim=0)
        static = static.unsqueeze(0)


        h3 = self.temporal_conv(static).squeeze()
        h3 = F.hardswish(h3)
        h3 = self.lin1(h3)
        h3 = F.hardswish(h3)
        h3 = self.lin2(h3)
        h3 = F.hardswish(h3)
        h3 = self.lin3(h3)
        return h3
    


class IceSheetModel_Multibranch_ablation1(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial_conv = SAGEConv(42, args.DIMENSIONALITIES[0])
        self.temporal_conv = TemporalConv(in_channels=8, out_channels=args.DIMENSIONALITIES[0], kernel_size=5)
        
        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))  # scalar between 0 and 1

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)


    def forward(self, graph_collection):

        h2 = None
        static = []

        for i in range(self.LAYER_FEATURE_COUNT):   
            e = graph_collection[i].x[:, 2:].float()

            static.append(e)

        e = torch.cat((graph_collection[0].x[:, :], graph_collection[1].x[:, 2:], graph_collection[2].x[:, 2:], graph_collection[3].x[:, 2:], graph_collection[4].x[:, 2:]), dim=1 )
        h2 = self.spatial_conv(e.float(), graph_collection[0].edge_index)
        
        static = torch.stack(static, dim=0)
        static = static.unsqueeze(0)
        h3 = self.temporal_conv(static).squeeze()

        assert h2.shape == h3.shape, f"Shape mismatch: h2 {h2.shape}, h3 {h3.shape}"

        h = h2 * self.alpha + h3 * (1 - self.alpha)
        
        h = F.hardswish(h)
        h = self.lin1(h)
        h = F.hardswish(h)
        h = self.lin2(h)
        h = F.hardswish(h)
        h = self.lin3(h)
        return h


class IceSheetModel_Multibranch_ablation1_NonAdaptive(IceSheetModel_Multibranch_ablation1_Old):
    '''
    Explicit alias for the non-adaptive Ablation1 variant.
    This path concatenates the spatial and temporal branches.
    '''
    pass
