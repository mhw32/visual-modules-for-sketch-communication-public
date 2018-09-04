r"""Main script to train models."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

from model import AdaptorHigh   # best FC6 model
from model import AdaptorMid    # best CONV42 model
from model import AdaptorEarly  # best POOL1 model
from dataset import VisualDataset
from globals import TRAIN_TEST_DIR


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    r"""Save a snapshot of the model.

    @param state: torch state_dict object
    @param is_best: boolean
                    whether this is our best performing model yet
    @param folder: string [default: ./]
                   which folder to save to
    @param filename: string [default: checkpoint.pth.tar]
                     name of checkpoint
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_checkpoint(file_path, use_cuda=False):
    r"""Given a saved checkpoint (see save_checkpoint function), we initialize
    a model from the saved weights.

    @param file_path: string
                      path to saved model path
    @param use_cuda: boolean [default: False]
                     whether parameters are by default CUDA casted
    """
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)

    if checkpoint['adaptor'] == 'high':
        assert 'hiddens_dim' in checkpoint
        model = AdaptorHigh(hiddens_dim=checkpoint['hiddens_dim'])
    elif checkpoint['adaptor'] == 'mid':
        model = AdaptorMid()
    elif checkpoint['adaptor'] == 'early':
        model = AdaptorLow()
    else:
        raise Exception('Unrecognized adaptor: %s' % checkpoint['adaptor'])

    model.load_state_dict(checkpoint['state_dict'])
    model.adaptor = checkpoint['adaptor']

    return model


class AverageMeter(object):
    r"""Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cross_entropy(input, target, eps=1e-6):
    r"""k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size N x K)
    @param target: torch.Tensor (size N x K)
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor (size N)
    """
    if not (target.size(0) == input.size(0)):
        raise ValueError(
            "Target size ({}) must be the same as input size ({})".format(
                target.size(0), input.size(0)))

    log_input = F.log_softmax(input + eps, dim=1)
    loss = target * log_input
    return -torch.mean(torch.sum(loss, dim=1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('adaptor', type=str, help='high|mid|low')
    # we pick 128 as the default b/c it seems the smallest dimensionality as which we can
    # still get good performance.
    parser.add_argument('--hiddens-dim', type=int, default=128, help='number of hidden dims [default: 128]')
    # this multiplier is helpful to keep gradients large but training stable.
    parser.add_argument('--loss-scale', type=float, default=10000., help='multiplier for loss [default: 10000.]')
    # by default, we use the first split
    parser.add_argument('--train-test-split-dir', type=str, default=os.path.join(TRAIN_TEST_DIR, '1'),
                        help='where to load train_test_split paths [default: ./train_test_split/1]')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='where to save checkpoints [./trained_models]')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='number of examples in a mini-batch [default: 10]')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate [default: 1e-4]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 100]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    train_dataset = VisualDataset(adaptor=args.adaptor, split='train', average_labels=True,
                                  train_test_split_dir=args.train_test_split_dir)
    val_dataset = VisualDataset(adaptor=args.adaptor, split='val', average_labels=True,
                                train_test_split_dir=args.train_test_split_dir)
    test_dataset = VisualDataset(adaptor=args.adaptor, split='test', average_labels=True,
                                 train_test_split_dir=args.train_test_split_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.adaptor == 'high':
        model = AdaptorHigh(args.hiddens_dim)
    elif args.adaptor == 'mid':
        model = AdaptorMid()
    elif args.adaptor == 'early':
        model = AdaptorEarly()
    else:
        raise Exception('Unrecognized adaptor: %s' % args.adaptor)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()

        for batch_idx, (sketch, label) in enumerate(train_loader):
            sketch = Variable(sketch)
            label = Variable(label.float())
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            # set optimizer defaults to 0
            optimizer.zero_grad()

            pred_logits = []
            photo_generator = train_dataset.gen_photos()
            for photo in photo_generator():
                photo = Variable(photo)
                if args.cuda:
                    photo = photo.cuda()
                photo = (photo.repeat(batch_size, 1) if args.adaptor == 'high' else
                         photo.repeat(batch_size, 1, 1, 1))
                pred_logit = model(photo, sketch)
                pred_logits.append(pred_logit)

            pred_logits = torch.cat(pred_logits, dim=1)
            loss = args.loss_scale * cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            loss.backward()
            optimizer.step()
            mean_grads = torch.mean(torch.cat([param.grad.cpu().data.contiguous().view(-1)
                                               for param in model.parameters()]))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t|Grad|: {:6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),  100. * batch_idx / len(train_loader),
                loss_meter.avg, mean_grads))

        print('====> Epoch: {}\tLoss: {:.4f}'.format(epoch, loss_meter.avg))

        return loss_meter.avg

    def validate():
        model.eval()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(val_loader))

        for batch_idx, (sketch, label) in enumerate(val_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label.float(), requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = []
            photo_generator = val_dataset.gen_photos()
            for photo in photo_generator():
                photo = Variable(photo)
                if args.cuda:
                    photo = photo.cuda()
                photo = (photo.repeat(batch_size, 1) if args.adaptor == 'high' else
                         photo.repeat(batch_size, 1, 1, 1))
                pred_logit = model(photo, sketch)
                pred_logits.append(pred_logit)

            pred_logits = torch.cat(pred_logits, dim=1)
            loss = args.loss_scale * cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            pbar.update()
        pbar.close()

        print('====> Val Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg


    def test():
        model.eval()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(test_loader))

        for batch_idx, (sketch, label) in enumerate(test_loader):
            sketch = Variable(sketch, volatile=True)
            label = Variable(label.float(), requires_grad=False)
            batch_size = len(sketch)

            if args.cuda:
                sketch = sketch.cuda()
                label = label.cuda()

            pred_logits = []
            photo_generator = test_dataset.gen_photos()
            for photo in photo_generator():
                photo = Variable(photo)
                if args.cuda:
                    photo = photo.cuda()
                photo = (photo.repeat(batch_size, 1) if args.adaptor == 'high' else
                         photo.repeat(batch_size, 1, 1, 1))
                pred_logit = model(photo, sketch)
                pred_logits.append(pred_logit)

            pred_logits = torch.cat(pred_logits, dim=1)
            loss = args.loss_scale * cross_entropy(pred_logits, label)
            loss_meter.update(loss.data[0], batch_size)

            pbar.update()
        pbar.close()

        print('====> Test Loss: {:.4f}'.format(loss_meter.avg))
        return loss_meter.avg


    best_loss = sys.maxint
    store_loss = np.zeros((args.epochs, 3))

    for epoch in xrange(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = validate()
        test_loss = test()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'adaptor': args.adaptor,
            'hiddens_dim': args.hiddens_dim,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder=args.out_dir)

        store_loss[epoch - 1, 0] = train_loss
        store_loss[epoch - 1, 1] = val_loss
        store_loss[epoch  - 1, 2] = test_loss

        np.save(os.path.join(args.out_dir, 'summary-loss.npy'), store_loss)
