from data_processing import DataPreprocessor, Delexicalizer
from slot_aligner import SlotAligner
import re
from nltk.tokenize import word_tokenize
import data_augmentation
import json
import pandas as pd
from alignment_utils import tokenize_mr
from data_loader import NMTDataset

# Use that if you want to preprocess the raw data and prepare a dataframe
# processor = DataPreprocessor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
#  	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])

# processor.save_data("data/inp_and_gt_name_near_food_no_inform.csv")


dataset = NMTDataset.load_dataset_and_make_vectorizer("data/inp_and_gt_name_near_food_no_inform.csv")

