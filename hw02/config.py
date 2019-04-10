import os

TRAIN_TPL = "dataset/2017_English_final/GOLD/Subtask_A/twitter-{year}train-A.txt"
TEST_TPL = "dataset/2017_English_final/GOLD/Subtask_A/twitter-{year}test-A.txt"
DEVTEST_TPL = "dataset/2017_English_final/GOLD/Subtask_A/twitter-{year}devtest-A.txt"

POS_FILE = "./dataset/train/pos.txt"
NEG_FILE = "./dataset/train/neg.txt"
NEU_FILE = "./dataset/train/neu.txt"


POS_TEST_FILE = "./dataset/test/pos.txt"
NEG_TEST_FILE = "./dataset/test/neg.txt"
NEU_TEST_FILE = "./dataset/test/neu.txt"

_dirname = os.path.dirname(__file__)
GLOVE_WORD2VEC = os.path.join(_dirname, "models/glove.6B.50d.txt")


# MODEL PARAM
MAX_SEQ_LENGTH = 30
EMBEDDING_DIM = 50
DEV_SIZE = .10
FILTER_SIZE = [3, 4, 5]
NUM_FILTERS = 32
L2_REG_LAMBDA = 0.0
DROPOUT_PROB = 0.5

# Training
BATCH_SIZE = 500
NUM_EPOCHS = 10
EVALUATE_EVERY = 100
CHECKPOINT_EVERY = 100000
NUM_CHECKPOINTS = 0