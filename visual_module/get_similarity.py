r"""Given a trained model (see train.py), generate a JSON dump of
similarity scores between sketches and photos.

Executing this file will produce two outputs:

    1) dump.json: a dictionary of photo and sketch name pairs to similarity
    2) dump-paths.json: paths to photos/sketches referenced in dump.json
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import json
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import ExhaustiveDataset
from train import load_checkpoint
from globals import TRAIN_TEST_DIR


def photo_uname(path):
    path = os.path.splitext(os.path.basename(path))[0]
    return path


def sketch_uname(path):
    path = '_'.join(os.path.splitext(os.path.basename(path))[0].split('_')[1:])
    path = path.split('-')[-1]
    path = path.replace('_trial', '')
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('--train-test-split-dir', type=str, default=os.path.join(TRAIN_TEST_DIR, '1'),
                        help='where to load train_test_split paths [default: ./train_test_split/1]')
    parser.add_argument('--out-dir', type=str, default='./',
                        help='where to dump files [default: ./]')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    model = load_checkpoint(args.model_path, use_cuda=args.cuda)
    model = model.eval()
    if model.cuda:
        model = model.cuda()

    dataset = ExhaustiveDataset(adaptor=model.adaptor, split='test',
                                train_test_split_dir=args.train_test_split_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    dist_jsons = defaultdict(lambda: {})
    pbar = tqdm(total=len(loader))
    test_sketchpaths = []

    for batch_idx, (sketch, sketch_object, sketch_context, sketch_path) in enumerate(loader):
        sketch_name = sketch_uname(sketch_path[0])
        test_sketchpaths.append(os.path.basename(sketch_path[0]))
        sketch = Variable(sketch, volatile=True)
        if args.cuda:
            sketch = sketch.cuda()
        photo_generator = dataset.gen_photos()

        for photo, photo_object, photo_path in photo_generator():
            photo_name = photo_uname(photo_path)
            photo = Variable(photo, volatile=True)
            batch_size = len(sketch)
            if args.cuda:
                photo = photo.cuda()

            pred = model(photo, sketch).squeeze(1).cpu().data[0]
            dist_jsons[photo_name][sketch_name] = float(pred)

        pbar.update()
    pbar.close()

    with open(os.path.join(args.out_dir, 'dump.json'), 'w') as fp:
        json.dump(dist_jsons, fp)

    with open(os.path.join(args.out_dir, 'dump-paths.json'), 'w') as fp:
        json.dump(test_sketchpaths, fp)
