r"""Make 5 separate train test splits. The ones used in
our experiments are provided in ./train_test_split folder.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil
import numpy as np
from dataset import VisualDataset
from globals import TRAIN_TEST_DIR

if __name__ == "__main__":
    # define a list of random seeds
    # this will generate the cross-validation we seek
    random_seeds = np.random.randint(0, 1000, size=5)
    for i in xrange(5):
                             # this could be set to anything, doesn't matter
        dset = VisualDataset(adaptor='high', split='train', average_labels=False,
                             photo_transform=None, sketch_transform=None,
                             train_test_split_dir=os.path.join(TRAIN_TEST_DIR, '%d' % (i + 1)),
                             random_seed=random_seeds[i])

