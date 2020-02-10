from data_processing import DataPreprocessor, Delexicalizer
from slot_aligner import SlotAligner
import re
from nltk.tokenize import word_tokenize
import data_augmentation
import json
import pandas as pd
from alignment_utils import tokenize_mr
from data_loader import NMTDataset, generate_nmt_batches

# Use that if you want to preprocess the raw data and prepare a dataframe
# processor = DataPreprocessor.from_files(train_input_path="data/e2e-dataset/trainset.csv", validation_input_path="data/e2e-dataset/devset.csv", 
#  	test_input_path="data/e2e-dataset/testset.csv", delexicalization_type="partial", delexicalization_slots=["name", "near", "food"])

# processor.save_data("data/inp_and_gt_name_near_food_no_inform.csv")

def test():

	dataset = NMTDataset.load_dataset_and_make_vectorizer("data/inp_and_gt_name_near_food_no_inform.csv")
	vect = dataset.get_vectorizer()

	print(vect.target_vocab.lookup_index(3))
	print(vect.target_vocab.lookup_index(2949))
	# for i in [2,     4,    34,    48,    16,    40,    39,    34,    81,    82,
 #           13,    18]:
	# 	print(vect.target_vocab.lookup_index(i))
	# sample = "x-name is a coffee shop along the river near x-near . the prices are quite high while the customer ratings are quite low . it is not recommended to take children there . "
	# vect._get_target_indices(sample)
	# batch_generator = generate_nmt_batches(dataset, 
	#                                        	       batch_size=1, 
	#                                                device="cpu")

	# for batch_index, batch_dict in enumerate(batch_generator):
	# 	print(batch_dict)
	# 	break
	# print("s")
	# sample = yield batch_generator
	# print(sample)
	

test()