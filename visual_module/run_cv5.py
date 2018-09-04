r"""Loop through the 5 splits and train a model for each."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import subprocess
from globals import TRAIN_TEST_DIR

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('adaptor', type=str, help='high|mid|early')
    parser.add_argument('--out-dir', type=str, default='./trained_models',
                        help='root folder for where to save trained models [default: ./trained_models]')
    parser.add_argument('--cuda-device', type=int, default=0, help='0|1|2|3 [default: 0]')
    args = parser.parse_args()

    for i in xrange(5):
        out_dir = '%s/%s/%d' % (args.out_dir, args.layer, i + 1)
        train_test_split_dir = os.path.join(TRAIN_TEST_DIR, '%d' % (i + 1))
        command = 'CUDA_VISIBLE_DEVICES={device} python train.py {layer} --train-test-split-dir {split_dir} --out-dir {out_dir} --cuda'.format(
            device=args.cuda_device, layer=args.layer, split_dir=train_test_split_dir, out_dir=out_dir)

        # hacky way to kick off a job
        subprocess.call(command, shell=True)
