r"""Loop through the 5 splits and run `get_similarity.py` for each model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess
from globals import TRAIN_TEST_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models_dir', type=str,
                        help='root where trained models are stored.')
    parser.add_argument('--out-dir', type=str, default='./similarity_dump',
                        help='root folder for where to save trained models [default: ./similarity_dump]')
    parser.add_argument('--cuda-device', type=int, default=0, help='0|1|2|3 [default: 0]')
    args = parser.parse_args()

    for layer in ['high', 'mid', 'early']:
        for i in xrange(5):
            print('Dumping files for (%s|%d)' % (layer, i))

            model_path = '%s/%s/%d/model_best.pth.tar' % (args.models_dir, layer, i + 1)
            out_dir = '%s/%s/%d/' % (args.out_dir, layer, i + 1)

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            split_dir = os.path.join(TRAIN_TEST_DIR, '%d' % (i + 1))
            command = 'CUDA_VISIBLE_DEVICES={device} python get_similarity.py {model} --train-test-split-dir {splitdir} --out-dir {outdir} --cuda'.format(
                device=args.cuda_device, model=model_path, splitdir=split_dir, outdir=out_dir)

            # call bash to run everything
            subprocess.call(command, shell=True)
