import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
from vocabulary import SequenceVocabulary
import time

def find_pair(seq, a, b):
    i = 1
    while i < len(seq):
        if seq[i] == b and seq[i - 1] == a: return i - 1
        i += 2 - (seq[i] == a)

class NMTVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""        
    def __init__(self, source_vocab, target_vocab, max_source_length, max_target_length):
        """
        Args:
            source_vocab (SequenceVocabulary): maps source words to integers
            target_vocab (SequenceVocabulary): maps target words to integers
            max_source_length (int): the longest sequence in the source dataset
            max_target_length (int): the longest sequence in the target dataset
        """
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        

    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """Vectorize the provided indices
        
        Args:
            indices (list): a list of integers that represent a sequence
            vector_length (int): an argument for forcing the length of index vector
            mask_index (int): the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)
        
        vector = np.zeros(vector_length, dtype=np.int64)

        #TODO: TEMPORARY WORKAROUND
        #MUST BE FIXED BY ITERATING OVER THE VALIDATION SET WHEN DEFINING THE MAX LENGTH

        # TODO This function should be changed, so Bigrams and Trigrams are included
        # The target vector should be a 2D array 3xVector_Length - further this will be passed to the model
        if len(indices) > vector_length:
            indices = indices[:vector_length]
            
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector

    def _vectorize_target(self, indices, vector_length, mask_index):
        if vector_length < 0:
            vector_length = len(indices)

        # Same workaround
        if len(indices) > vector_length:
            indices = indices[:vector_length]

        vocabulary_length = len(self.target_vocab)

        # start = time.time()

        # The target vocabulary has shape: NUMBER_WORDS x VOCAB_SIZE
        # Those are the target vectors for each one of the words in the sentence
        vector = np.zeros((vector_length, vocabulary_length), dtype=np.double)
        
        # Iterate over the indices
        for word_idx, row in enumerate(indices):
            # This is the index target word which should be generated
            target_word_idx = row[0]

            # It gets 0.5 of the total probability
            vector[word_idx][target_word_idx] = 0.5

            # Calculate how to distribute the rest probabilities amongst the bigrams
            # All future n_grams
            future_ngrams_len = len(row[1:])

            if future_ngrams_len == 0:
                future_ngrams_len = 1
            # Distribute the rest probability mass amognst the future ngrams that may appear
            future_ngram_prob = 0.5*(1./future_ngrams_len)

            for idx_1, ngram_idx in enumerate(row[1:]):
                vector[word_idx][ngram_idx] = future_ngram_prob

        # Fill with the rest with the mask index
        for word_idx in range(len(indices), vector_length):
            vector[word_idx][mask_index] = 1. 

        # end = time.time()
        # print(end - start)

        return vector

    def _vectorize_target_multitask(self, indices, vector_length, mask_index):

        if vector_length < 0:
            vector_length = len(indices)

        # Same workaround
        if len(indices) > vector_length:
            indices = indices[:vector_length]

        vocabulary_length = len(self.target_vocab)

        index_first_bigram = self.target_vocab.lookup_token(self.target_vocab.bigrams[0])

        # The target vocabulary has shape: NUMBER_WORDS x VOCAB_SIZE
        # Those are the target vectors for each one of the words in the sentence

        # One vector for unigrams
        vector_unigrams = np.zeros((vector_length, index_first_bigram), dtype=np.double)

        # One vector for bigrams that will contain the futire bigrams at each position
        vector_bigrams = np.zeros((vector_length, vocabulary_length-index_first_bigram), dtype=np.double)

        # Iterate over the indices
        for word_idx, row in enumerate(indices):
            # This is the index of the target unigram which should be generated
            target_word_idx = row[0]
            vector_unigrams[word_idx][target_word_idx] = 1.

            # Calculate how to distribute the rest probs amongst the bigrams

            # Get all future bigrams
            future_ngrams_len = len(row[1:])

            if future_ngrams_len == 0:
                future_ngrams_len = 1
            
            # Distribute the probability mass amongst the future ngrams that may appear
            future_ngram_prob = 1./future_ngrams_len

            for idx_1, ngram_idx in enumerate(row[1:]):
                vector_bigrams[word_idx][ngram_idx-index_first_bigram] = future_ngram_prob

        # Fill with the rest with the mask index
        for word_idx in range(len(indices), vector_length):
            vector_unigrams[word_idx][mask_index] = 1.
            vector_bigrams[word_idx][mask_index] = 1.
        
        return vector_unigrams, vector_bigrams
        
    def _get_source_indices(self, text):
        """Return the vectorized source text
        
        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            indices (list): list of integers representing the text
        """
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(self.source_vocab.lookup_token(token) for token in text.split(" "))
        indices.append(self.source_vocab.end_seq_index)
        return indices
    
    def _get_target_indices(self, text):
        """Return the vectorized source text
        
        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list): list of integers representing the observations in target decoder 
                y_indices (list): list of integers representing predictions in target decoder
        """

        # TODO Some changes should be made here, so the whole thing operates on Bigrams and Trigrams
        # Several target indices should be returned for each one of the words
        #start = time.time()

        all_target_tokens = text.split(" ")

        # Prune the bigrams
        included_bigrams = []
        for bigram in self.target_vocab.bigrams:
            exists = bigram.split(" ") in [all_target_tokens[i:i+2] for i in range(len(all_target_tokens) - 1)]
            if exists:
                included_bigrams.append(bigram)

        all_indices = []
        for idx, token in enumerate(all_target_tokens):
            # print(token)
            # print("------------------------------")
            future_tokens = all_target_tokens[idx:]
            future_bigrams = []
            for bigram in included_bigrams:
                exists = bigram.split(" ") in [future_tokens[i:i+2] for i in range(len(future_tokens) - 1)]
                if exists:
                    future_bigrams.append(self.target_vocab.lookup_token(bigram))

            all_indices.append([self.target_vocab.lookup_token(token)]+future_bigrams)
                    # print(bigram)
                    # print(self.target_vocab.lookup_token(bigram))
            # print("##############################")
                # if bigram in " ".join(all_target_tokens[idx:]):
                #     print(bigram)

        indices = [self.target_vocab.lookup_token(token) for token in text.split(" ")]
        x_indices = [self.target_vocab.begin_seq_index] + indices

        # The target indices will be changed because n grams are included
        #y_indices = indices + [self.target_vocab.end_seq_index]

        y_indices = [[self.target_vocab.begin_seq_index]] + all_indices
        #end = time.time()
        #print("Prepare target indices time: ", (end - start))

        return x_indices, y_indices


    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        """Return the vectorized source and target text
        
        The vetorized source text is just the a single vector.
        The vectorized target text is split into two vectors in a similar style to 
            the surname modeling in Chapter 7.
        At each timestep, the first vector is the observation and the second vector is the target. 
        
        
        Args:
            source_text (str): text from the source language
            target_text (str): text from the target language
            use_dataset_max_lengths (bool): whether to use the global max vector lengths
        Returns:
            The vectorized data point as a dictionary with the keys: 
                source_vector, target_x_vector, target_y_vector, source_length
        """
        source_vector_length = -1
        target_vector_length = -1
        
        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1
            
        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(source_indices, 
                                        vector_length=source_vector_length, 
                                        mask_index=self.source_vocab.mask_index)
        
        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)

        # target_y_vector = self._vectorize_target(target_y_indices,
        #                                 vector_length=target_vector_length,
        #                                 mask_index=self.target_vocab.mask_index)

        # Just a test for now
        target_unigrams, target_bigrams = self._vectorize_target_multitask(target_y_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)


        # target_y_vector = self._vectorize(target_y_indices,
        #                                 vector_length=target_vector_length,
        #                                 mask_index=self.target_vocab.mask_index)
        return {"source_vector": source_vector, 
                "target_x_vector": target_x_vector, 
                "target_unigrams_vector": target_unigrams,
                "target_bigrams_vector": target_bigrams,
                "source_length": len(source_indices)}
        
    @classmethod
    def from_dataframe(cls, bitext_df):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            bitext_df (pandas.DataFrame): the parallel text dataset
        Returns:
            an instance of the NMTVectorizer
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()
        
        max_source_length = 0
        max_target_length = 0

        for idx, row in bitext_df.iterrows():
            if type(row["source_language"]) != str:
                print(row["target_language"])
                print(idx)
                continue

            source_tokens = row["source_language"].split(" ")
            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)
            for token in source_tokens:
                source_vocab.add_token(token)
            
            # TODO Change the code, so it loads adds also bigrams and trigrams to the dictionary
            target_tokens = row["target_language"].split(" ")
            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)
            for token in target_tokens:
                target_vocab.add_token(token)
            
        target_vocab.augment_with_ngrams(bitext_df, 10)

        print(str(target_vocab))

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])
        
        return cls(source_vocab=source_vocab, 
                   target_vocab=target_vocab, 
                   max_source_length=contents["max_source_length"], 
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {"source_vocab": self.source_vocab.to_serializable(), 
                "target_vocab": self.target_vocab.to_serializable(), 
                "max_source_length": self.max_source_length,
                "max_target_length": self.max_target_length}
