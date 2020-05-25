import logging
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Collection, Dict

from dgl.nn import SAGEConv
from ogb.linkproppred import DglLinkPropPredDataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import dgl
import click


logger = logging.getLogger(__name__)


@dataclass
class Batch:
    blocks: List[dgl.DGLHeteroGraph]
    decode_graph: dgl.DGLHeteroGraph
    input_features: Tensor
    labels: Tensor = None


class SageNet(torch.nn.Module):

    def __init__(self, n_hops: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self._n_hops = n_hops
        self._sage_layers = torch.nn.ModuleList()
        self._activate = torch.nn.ReLU()

        for i in range(n_hops):
            sage_in_dim = input_dim if i == 0 else hidden_dim
            sage_out_dim = output_dim if i == (n_hops-1) else hidden_dim
            self._sage_layers.append(SAGEConv(
                in_feats=sage_in_dim, out_feats=sage_out_dim, aggregator_type="mean"))

    def forward(self, blocks: List[dgl.DGLHeteroGraph], input_features: Tensor) -> Tensor:
        h = input_features
        for i, (layer, block) in enumerate(zip(self._sage_layers, blocks)):
            h = layer(block, (h, h[:block.number_of_dst_nodes()]))
            if i < self._n_hops - 1:
                h = self._activate(h)
        return h


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decode_graph: dgl.DGLHeteroGraph, node_representations: Tensor):
        with decode_graph.local_scope():
            decode_graph.ndata["h"] = node_representations
            decode_graph.apply_edges(dgl.function.u_dot_v("h", "h", "logits"))
            return decode_graph.edata["logits"]


class SageLinkPrediction(torch.nn.Module):
    def __init__(self, encoder: SageNet, decoder: Decoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, blocks: List[dgl.DGLHeteroGraph], decode_graph: dgl.DGLHeteroGraph, input_feature: Tensor) \
            -> Tensor:
        h = self._encoder(blocks, input_feature)
        logits = self._decoder(decode_graph, h)
        return logits


class NeighborSampler:
    def __init__(self, g: dgl.DGLHeteroGraph, fanouts: List[int], negative_sampling: bool):
        self._g = g
        self._fanouts = fanouts
        self._negative_sampling = negative_sampling
        if negative_sampling is True:
            self._negative_weights = g.in_degrees().float() ** 0.75
        else:
            self._negative_weights = None

    def _sample_negative_labels(self, labels: Tensor) -> Tensor:
        head = labels[:, 0]
        neg_tail = self._negative_weights.multinomial(len(head), replacement=True)
        negative_labels = torch.cat([head.unsqueeze(1), neg_tail.unsqueeze(1), torch.zeros(len(head), 1).long()], dim=1)
        return negative_labels

    def sample(self, labels: Tensor) -> Batch:
        """sample blocks and decode graph from labels

        :param labels: tensor of dim Nx3, (src_nodes, dst_nodes, labels)
        :return: blocks for calculating node representations, a decode graph, and edge labels
        """
        labels = torch.cat([label[0].unsqueeze(0) for label in labels], dim=0)
        if self._negative_sampling is True:
            negative_labels = self._sample_negative_labels(labels)
            labels = torch.cat([labels, negative_labels], dim=0)

        decode_graph = dgl.graph(data=(labels[:, 0], labels[:, 1]), num_nodes=self._g.number_of_nodes())
        decode_graph = dgl.compact_graphs(decode_graph)
        seed_nodes = decode_graph.ndata[dgl.NID]

        blocks = list()
        for fanout in self._fanouts:
            sub_graph = dgl.sampling.sample_neighbors(g=self._g, nodes=seed_nodes, fanout=fanout)
            block = dgl.to_block(sub_graph, seed_nodes)
            blocks.insert(0, block)
            seed_nodes = block.srcdata[dgl.NID]
        input_features = self._g.ndata["feat"][blocks[0].srcdata[dgl.NID]]
        return Batch(blocks, decode_graph, input_features, labels[:, 2])


class SageLinkPredictionTrainer:

    def __init__(self, model: SageLinkPrediction, device: str):
        self._model = model.to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        self._loss = torch.nn.BCEWithLogitsLoss()
        self._epoch: int = 0
        self._device = device

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _train_batch(self, batch: Batch) -> Tuple[float, float]:
        self._model.train()
        self._optimizer.zero_grad()
        labels = batch.labels.to(self._device)
        logits = self._model(batch.blocks, batch.decode_graph, batch.input_features.to(self._device))
        loss = self._loss(logits, labels.float())
        loss.backward()
        self._optimizer.step()
        return loss.item(), self._get_accuracy(logits, labels)

    @torch.no_grad()
    def _eval_batch(self, batch) -> float:
        self._model.eval()
        labels = batch.labels.to(self._device)
        logits = self._model(batch.blocks, batch.decode_graph, batch.input_features.to(self._device))
        return self._get_accuracy(logits, labels)

    @staticmethod
    def _get_accuracy(logits: Tensor, labels: Tensor):
        pred = logits > 0.5
        return ((pred == labels).sum().float()/len(logits)).item()

    def train(self, train_batches: Iterable[Batch], validate_batches: Iterable[Batch],
              test_batches: Iterable[Batch]):
        for batch_index, batch in enumerate(train_batches):
            running_loss, running_accuracy = self._train_batch(batch)
            logger.info(f"epoch: {self._epoch}; batch {batch_index}; train_loss: {running_loss}; "
                        f"train_acc: {running_accuracy};")
        validate_accuracies = [self._eval_batch(batch) for batch in validate_batches]
        test_accuracies = [self._eval_batch(batch) for batch in test_batches]

        validate_accuracy = sum(validate_accuracies)/len(validate_accuracies)
        test_accuracy = sum(test_accuracies)/len(test_accuracies)
        logger.info(f"epoch: {self._epoch}; validate_acc: {validate_accuracy}; test_acc: {test_accuracy}")


def prepare_train_labels() -> Tuple[dgl.DGLHeteroGraph, Tensor, Tensor, Tensor]:
    dataset = DglLinkPropPredDataset(name="ogbl-collab")
    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    graph: dgl.DGLGraph = dataset[0]

    train_src_nodes = torch.cat([train_edge["edge"][:, 0], train_edge["edge"][:, 1]], dim=0)
    train_dst_nodes = torch.cat([train_edge["edge"][:, 1], train_edge["edge"][:, 0]], dim=0)
    train_graph = dgl.graph(data=(train_src_nodes, train_dst_nodes), num_nodes=graph.number_of_nodes())

    train_graph.ndata["feat"] = graph.ndata["feat"]

    train_labels = train_edge["edge"]
    train_labels = torch.cat([train_labels, torch.ones(len(train_labels), 1).long()], dim=1)
    valid_labels = get_label_from_split(valid_edge)
    test_labels = get_label_from_split(test_edge)
    return train_graph, train_labels, valid_labels, test_labels


def get_label_from_split(split_edge: Dict[str, Tensor]) -> Tensor:
    n_pos_labels, n_neg_labels = len(split_edge["edge"]), len(split_edge["edge_neg"])
    pos_labels = torch.cat([split_edge["edge"], torch.ones(n_pos_labels, 1).long()], dim=1)
    neg_labels = torch.cat([split_edge["edge_neg"], torch.zeros(n_neg_labels, 1).long()], dim=1)
    return torch.cat([pos_labels, neg_labels], dim=0)


@click.command("train link prediction model with GraphSage on ogbl-collab dataset")
@click.option("--n-epoch", type=int, default=1)
@click.option("--batch-size", type=int, default=10_000)
@click.option("--sample-fanout", type=int, default=20)
@click.option("--n-hops", type=int, default=2)
@click.option("--device", type=str, default="cpu")
def train(n_epoch: int, batch_size: int, sample_fanout: int, n_hops: int, device: str):

    logger.info("start to prepare data")
    train_graph, train_labels, valid_labels, test_labels = prepare_train_labels()

    logger.info("data prepared")

    encoder = SageNet(n_hops=n_hops, input_dim=train_graph.ndata["feat"].shape[1], hidden_dim=64, output_dim=1)
    model = SageLinkPrediction(encoder=encoder, decoder=Decoder())

    train_sampler = NeighborSampler(train_graph, fanouts=[sample_fanout]*n_hops, negative_sampling=True)
    valid_sampler = NeighborSampler(train_graph, fanouts=[50]*n_hops, negative_sampling=False)
    test_sampler = NeighborSampler(train_graph, fanouts=[50]*n_hops, negative_sampling=False)

    trainer = SageLinkPredictionTrainer(model=model, device=device)

    train_data_loader = DataLoader(dataset=TensorDataset(train_labels), batch_size=batch_size,
                                   collate_fn=train_sampler.sample, shuffle=True)
    valid_data_loader = DataLoader(dataset=TensorDataset(valid_labels), batch_size=batch_size,
                                   collate_fn=valid_sampler.sample, shuffle=False)
    test_data_loader = DataLoader(dataset=TensorDataset(test_labels), batch_size=batch_size,
                                  collate_fn=test_sampler.sample, shuffle=False)

    logger.info("start to train")
    for epoch in range(n_epoch):
        trainer.set_epoch(epoch)
        trainer.train(train_data_loader, valid_data_loader, test_data_loader)
    logger.info("train complete")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
    train()
