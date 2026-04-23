"""Baseline GAT and GIN model definitions used for comparison experiments."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv


class IceSheetModel_Multibranch_ablation4_GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GATConv(
            42,
            args.DIMENSIONALITIES[0],
            heads=1,
            concat=False,
            edge_dim=1,
            bias=True,
        )

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

    def forward(self, graph_collection):
        h1 = None

        e = torch.cat(
            (
                graph_collection[0].x[:, :],
                graph_collection[1].x[:, 2:],
                graph_collection[2].x[:, 2:],
                graph_collection[3].x[:, 2:],
                graph_collection[4].x[:, 2:],
            ),
            dim=1,
        )

        edge_attr = graph_collection[0].edge_weight.float().view(-1, 1)
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, edge_attr=edge_attr)
        h1 = F.hardswish(h1)
        h1 = self.lin1(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin2(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin3(h1)
        return h1


class IceSheetModel_Multibranch_ablation4_GIN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(42, args.DIMENSIONALITIES[0]),
                torch.nn.Hardswish(),
                torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[0]),
            ),
            train_eps=True,
        )

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

    def forward(self, graph_collection):
        h1 = None

        e = torch.cat(
            (
                graph_collection[0].x[:, :],
                graph_collection[1].x[:, 2:],
                graph_collection[2].x[:, 2:],
                graph_collection[3].x[:, 2:],
                graph_collection[4].x[:, 2:],
            ),
            dim=1,
        )

        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index)
        h1 = F.hardswish(h1)
        h1 = self.lin1(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin2(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin3(h1)
        return h1


class IceSheetModel_Multibranch_ablation4_nonphy_GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GATConv(
            7,
            args.DIMENSIONALITIES[0],
            heads=1,
            concat=False,
            edge_dim=1,
            bias=True,
        )

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

    def forward(self, graph_collection):
        h1 = None

        e = torch.cat(
            (
                graph_collection[0].x[:, :],
                graph_collection[1].x[:, 2:],
                graph_collection[2].x[:, 2:],
                graph_collection[3].x[:, 2:],
                graph_collection[4].x[:, 2:],
            ),
            dim=1,
        )

        edge_attr = graph_collection[0].edge_weight.float().view(-1, 1)
        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index, edge_attr=edge_attr)
        h1 = F.hardswish(h1)
        h1 = self.lin1(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin2(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin3(h1)
        return h1


class IceSheetModel_Multibranch_ablation4_nonphy_GIN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spectral_conv = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(7, args.DIMENSIONALITIES[0]),
                torch.nn.Hardswish(),
                torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[0]),
            ),
            train_eps=True,
        )

        self.LAYER_FEATURE_COUNT = args.LAYER_FEATURE_COUNT
        self.ADAPTIVE = args.ADAPTIVE

        self.lin1 = torch.nn.Linear(args.DIMENSIONALITIES[0], args.DIMENSIONALITIES[1])
        self.lin2 = torch.nn.Linear(args.DIMENSIONALITIES[1], args.DIMENSIONALITIES[2])
        self.lin3 = torch.nn.Linear(args.DIMENSIONALITIES[2], args.LAYER_PREDICT_COUNT)

    def forward(self, graph_collection):
        h1 = None

        e = torch.cat(
            (
                graph_collection[0].x[:, :],
                graph_collection[1].x[:, 2:],
                graph_collection[2].x[:, 2:],
                graph_collection[3].x[:, 2:],
                graph_collection[4].x[:, 2:],
            ),
            dim=1,
        )

        h1 = self.spectral_conv(e.float(), graph_collection[0].edge_index)
        h1 = F.hardswish(h1)
        h1 = self.lin1(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin2(h1)
        h1 = F.hardswish(h1)
        h1 = self.lin3(h1)
        return h1
