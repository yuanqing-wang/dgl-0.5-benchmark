import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
import jax
from jax import numpy as jnp
import flax
import flax.nn as nn

from dgl.utils import expand_as_pair
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils import Logger

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)

    def apply(self, graph, feat, in_feats, out_feats):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src, feat_dst = feat
        else:
            feat_src = feat_dst = feat
        h_self = feat_dst
       
        if isinstance(feat_src, jax.interpreters.ad.JVPTracer):
            graph = graph.cpu()

        graph.srcdata['h'] = feat_src

        graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        h_neigh = graph.dstdata['neigh']
        rst = nn.Dense(h_self, out_feats) + nn.Dense(h_neigh, out_feats)

        return rst

class GraphSAGE(nn.Module):
    def apply(self, g, x, in_feats, hidden_feats, out_feats, num_layers, dropout):
        with nn.stochastic(jax.random.PRNGKey(0)):
            x = SAGEConv(g, x, in_feats, hidden_feats)

            for idx in range(num_layers-2):
                x = SAGEConv(g, x, hidden_feats, hidden_feats)
                x = nn.BatchNorm(x)
                x = nn.dropout(x, rate=dropout)

            x = SAGEConv(g, x, hidden_feats, out_feats)

        return jax.nn.log_softmax(x, axis=-1)

def train(model, g, feats, y_true, train_idx, optimizer):
    g = g.to(jax.devices('gpu')[0])
    def loss_fn(model, y_true=y_true):
        out = model(g, feats)[train_idx]
        y_true = y_true[train_idx].flatten()

        print(out.shape)
        print(y_true.shape)

        y_true = jax.nn.one_hot(y_true, 40)
        loss = jnp.mean(-out * y_true)
        return loss

    # grad = jax.jacfwd(loss_fn)(optimizer.target)
    # loss = loss_fn(optimizer.target)

    loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
    print(loss)

    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss

def test(model, g, feats, y_true, split_idx, evaluator):
    model.eval()

    out = model(g, feats)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GraphSAGE Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    g, labels = dataset[0]
    feats = jax.device_put(
            g.ndata['feat'],
            jax.devices("gpu")[0]
    )

    g = g.to(jax.devices("cpu")[0])

    g = dgl.to_bidirected(g)
    g = g.int()
    g = g.to(jax.devices("gpu")[0])

    train_idx = split_idx['train'].numpy()

    _model = GraphSAGE.partial(in_feats=feats.shape[-1],
                      hidden_feats=args.hidden_channels,
                      out_feats=dataset.num_classes,
                      num_layers=args.num_layers,
                      dropout=args.dropout)

    _, initial_params = _model.init(jax.random.PRNGKey(0), g, feats)
    model = nn.Model(_model, initial_params)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        _, initial_params = _model.init(jax.random.PRNGKey(0), g, feats)
        model = nn.Model(_model, initial_params)
        optimizer = flax.optim.Adam(args.lr).create(model)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            optimizer, loss = train(model, g, feats, labels, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))
            if not args.eval:
                continue

            result = test(model, g, feats, labels, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()


if __name__ == '__main__':
    main()
