r"""Helpful global variables used in the scripts."""

import os

SCRIPTS_DIR = os.path.realpath(os.path.dirname(__file__))
TRAIN_TEST_DIR = os.path.join(SCRIPTS_DIR, 'train_test_split')
DATA_DIR = os.path.join(SCRIPTS_DIR, 'data')
