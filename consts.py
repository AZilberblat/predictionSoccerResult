#   Dataset source: http://football-data.co.uk/englandm.php

TRAIN_DATASET_PATH = './dataset/train/soccer.csv'
TEST_DATASET_PATH = './dataset/test/soccer.csv'
RAW_DATA_DIR = './dataset/raw'
FIXED_DATA_FILENAME = 'soccer.csv'

CSV_SEPARATOR = ','

HOME_WIN = 0
DRAW = 1
AWAY_WIN = 2

NUM_CLASSES = 3

LABEL_INDEX = 3
NUMERICAL_DATA_INDEX = 5

NUMBER_OF_FEATURES = 16

MODEL_PATH = './sgd_model'

VALUE_TO_LABEL = {'H': HOME_WIN, 'A': AWAY_WIN, 'D': DRAW}

SELECTED_FEATURES = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST',
                    'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']


TRAIN_SET_SIZE_PERCENTAGE = 0.8
TEST_SET_SIZE_PERCENTAGE = 1.0 - TRAIN_SET_SIZE_PERCENTAGE

