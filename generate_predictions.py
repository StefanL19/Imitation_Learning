from encoders import NMTEncoder
from decoders import NMTDecoder
from data_loader import NMTDataset, generate_nmt_batches
from models import NMTModel
import sampler
import utils_inference
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from math import log
from nltk.translate import bleu_score
import seaborn as sns
import matplotlib.pyplot as plt
from alignment_utils import tokenize_mr, tokenize_mr_upper
from data_processing import Delexicalizer
from slot_aligner import SlotAligner
from alignment_utils import tokenize_mr, tokenize_mr_upper
import sampler
import numpy as np

args = Namespace(dataset_csv="data/inp_and_gt_name_near_food_no_inform.csv",
                 vectorizer_file="test.json",
                 model_state_file="test.pth",
                 save_dir="data/trained_models/",
                 cuda=True,
                 seed=1337,
                 batch_size=1,
                 source_embedding_size=48, 
                 target_embedding_size=48,
                 encoding_size=256,
                 results_save_path="data/results/res.csv",
                 results_gt_save_path="data/results/gt_ref.txt",
                 results_sampled_save_path="data/results/sampled_ref.txt")

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.cuda.manual_seed_all(args.seed)

args.device = torch.device("cuda" if args.cuda else "cpu")

dataset = NMTDataset.load_dataset_and_load_vectorizer(args.dataset_csv, args.save_dir+args.vectorizer_file)

vectorizer = dataset.get_vectorizer()

model = NMTModel(source_vocab_size=len(vectorizer.source_vocab), 
                 source_embedding_size=args.source_embedding_size, 
                 target_vocab_size=len(vectorizer.target_vocab),
                 target_embedding_size=args.target_embedding_size, 
                 encoding_size=args.encoding_size,
                 target_bos_index=vectorizer.target_vocab.begin_seq_index,
                 is_training=False,
                 attention_mode="bahdanau")

model.load_state_dict(torch.load(args.save_dir+args.model_state_file, map_location=torch.device(args.device)))
model.eval().to(args.device)

inference_sampler = sampler.NMTSampler(vectorizer, model, use_reranker=True, beam_width=10)
dataset.set_split('val')

batch_generator = generate_nmt_batches(dataset, 
                                       batch_size=args.batch_size, 
                                       device=args.device,
                                       shuffle=False)

all_results = []

val_bar = tqdm(desc='validation_res',
                        total=dataset.get_num_batches(args.batch_size), 
                        position=0, 
                        leave=True)

# Counters of total overgenerated and undergenerated slots
total_overgen = 0
total_undergen = 0

for batch_idx in range(0, dataset.get_num_batches(args.batch_size)):
    batch_dict = next(batch_generator)
    inference_sampler.apply_to_batch(batch_dict)
    
    for i in range(args.batch_size):
        res = inference_sampler.get_ith_item(i, False)
        total_overgen += res["overgenerated"]
        total_undergen += res["undergenerated"]
        all_results.append(res)

    val_bar.set_postfix(total_overgenerated=total_overgen, totral_undergenerated=total_undergen)
    val_bar.update()

res_mrs_gt = []
res_input_mr = []
res_sample_no_delex = []
res_ref_gt = []
res_sample_original = []


for result in all_results:
    # print("MR GT: ")
    # print(result["mrs_gt"])
    res_mrs_gt.append(result["mrs_gt"])
    # print("------------------")
    # print("Input MR: ")
    # print(result["source"])
    res_input_mr.append(result["source"])
    # print("-------------------------")
    # print("Reference ground truth: ")
    # print(result["reference_gt"])
    res_ref_gt.append(result["reference_gt"])
    # print("---------------------------")
    # print("Sampled no delexicalization: ")
    # print(result["sampled_normalized"])
    res_sample_no_delex.append(result["sampled_normalized"])
    # print("---------------------------")
    # print("Sampled with delexicalization: ")
    # print(" ".join(result["sampled"]))
    res_sample_original.append(result["sampled"])
    # print("---------------------------")
    # print("############################")

res_df = pd.DataFrame({'MR_GT': res_mrs_gt, 'Input_MR': res_input_mr,
         'Reference_Ground_Truth': res_ref_gt, 'Sample_No_Delex':res_sample_no_delex, 'Sample_Delex':res_sample_original})


res_df.to_csv(args.results_save_path, encoding='utf-8', index=False)

df = pd.read_csv(args.results_save_path)

grp = df.groupby(['MR_GT'])

for name, group in grp:
    with open(args.results_gt_save_path, "a") as f:
        for samp in group["Reference_Ground_Truth"]:
            f.write(samp)
            f.write("\n")

        f.write("\n")

    with open(args.results_sampled_save_path, "a") as f_s:
        f_s.write(group["Sample_No_Delex"].iloc[0])
        f_s.write("\n")


# #Contains the samples for the current batch
# batch_dict = next(batch_generator)

# batch_dict = next(batch_generator)

# batch_dict = next(batch_generator)

# batch_dict = next(batch_generator)

# batch_dict = next(batch_generator)

# batch_dict = next(batch_generator)


# inference_sampler.apply_to_batch(batch_dict)

# all_results = []

# for i in range(args.batch_size):
#     all_results.append(inference_sampler.get_ith_item(i, False))

# res_mrs_gt = []
# res_input_mr = []
# res_sample_no_delex = []
# res_ref_gt = []
# res_sample_original = []

# for result in all_results:
#     print("MR GT: ")
#     print(result["mrs_gt"])
#     res_mrs_gt.append(result["mrs_gt"])
#     print("------------------")
#     print("Input MR: ")
#     print(result["source"])
#     res_input_mr.append(result["source"])
#     print("-------------------------")
#     print("Reference ground truth: ")
#     print(result["reference_gt"])
#     res_ref_gt.append(result["reference_gt"])
#     print("---------------------------")
#     print("Sampled no delexicalization: ")
#     print(result["sampled_normalized"])
#     res_sample_no_delex.append(result["sampled_normalized"])
#     print("---------------------------")
#     print("Sampled with delexicalization: ")
#     print(" ".join(result["sampled"]))
#     res_sample_original.append(result["sampled"])
#     print("---------------------------")
#     print("############################")

# res_df = pd.DataFrame({'MR_GT': res_mrs_gt, 'Input_MR': res_input_mr,
#          'Reference_Ground_Truth': res_ref_gt, 'Sample_No_Delex':res_sample_no_delex, 'Sample_Delex':res_sample_original})

# res_df.to_csv("data/res.csv", encoding='utf-8', index=False)
