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
from sparsemax import Sparsemax

def verbose_attention(encoder_state_vectors, query_vector):
    """A descriptive version of the neural attention mechanism 
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state in decoder GRU
    Returns:
        
    """

    print("The shape of the encoder state vectors is: ", encoder_state_vectors.shape)
    print("The shape of the query vector is: ", query_vector.shape)

    batch_size, num_vectors, vector_size = encoder_state_vectors.size()

    # APPLYING DOT PRODUCT ATTENTION
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size), 
                              dim=2)

    # SOFTMAX OVER THE SCORES
    vector_probabilities = F.softmax(vector_scores, dim=1)
    
    # WEIGHT THE VECTORS WITH THE CORRESPONDING SCORES
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    
    # CALCULATE THE CONTEXT VECTORS FOR EACH OF THE BATCH ELEMENTS BY SUMMING THE WEIGHTED VECTORS 
    context_vectors = torch.sum(weighted_vectors, dim=1)

    return context_vectors, vector_probabilities, vector_scores


def terse_attention(encoder_state_vectors, query_vector):
    """A shorter and more optimized version of the neural attention mechanism
    
    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1), 
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities

class NMTDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index, attention_mechanism, training_mode=False):
        """
        Args:
            num_embeddings (int): number of embeddings is also the number of 
                unique words in target vocabulary 
            embedding_size (int): the embedding vector size
            rnn_hidden_size (int): size of the hidden rnn state
            bos_index(int): begin-of-sequence index
        """
        super(NMTDecoder, self).__init__()
        self._rnn_hidden_size = rnn_hidden_size
        self.target_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_size, 
                                             padding_idx=0)
        self.target_embedding = self.target_embedding.detach()
        
        self.gru_cell = nn.GRUCell(embedding_size + rnn_hidden_size, 
                                   rnn_hidden_size)
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)

        ## If we feed the context vectors as Luong et al. does
        # self.classifier = nn.Linear(rnn_hidden_size * 2, num_embeddings)

        ## If we do not feed the context vectors as Bahdanau et al. does
        # self.classifier = nn.Linear(rnn_hidden_size, num_embeddings)

        ## If we apply factorization
        # The idea is that we will generate an embedding vector and this embedding vector
        # will be compared to all embedding vectors in the target vocabulary
        # the one with the most similarity will be returned
        self.classifier = nn.Linear(rnn_hidden_size, embedding_size)

        self.bos_index = bos_index
        self._sampling_temperature = 3
        self.training_mode = training_mode
        self.attention_mechanism = attention_mechanism
        self.num_embeddings = num_embeddings

        print("The training mode is: ", training_mode)
    
    def _init_indices(self, batch_size):
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    def _init_context_vectors(self, batch_size):
        """ return a zeros vector for initializing the context """
        return torch.zeros(batch_size, self._rnn_hidden_size)
            
    def forward(self, encoder_state, initial_hidden_state, target_sequence, sample_probability=0.0):
        """The forward pass of the model
        
        Args:
            encoder_state (torch.Tensor): the output of the NMTEncoder
            initial_hidden_state (torch.Tensor): The last hidden state in the  NMTEncoder
            target_sequence (torch.Tensor): the target text data tensor
            sample_probability (float): the schedule sampling parameter
                probabilty of using model's predictions at each decoder step
        Returns:
            output_vectors (torch.Tensor): prediction vectors at each output step
        """
        if target_sequence is None:
            sample_probability = 1.0
        else:
            # We are making an assumption there: The batch is on first
            # The input is (Batch, Seq)
            # We want to iterate over sequence so we permute it to (S, B)
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)

        # use the provided encoder hidden state as the initial hidden state
        h_t = self.hidden_map(initial_hidden_state)

        
        batch_size = encoder_state.size(0)

        # initialize context vectors to zeros
        context_vectors = self._init_context_vectors(batch_size)
        
        # initialize first y_t word as BOS
        y_t_index = self._init_indices(batch_size)
        
        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()
        
        # Initial Dirichlet alpha
        attention_energies = torch.tensor(np.zeros((encoder_state.size()[0],  encoder_state.size()[1]), dtype=np.float32), requires_grad=False)
        attention_energies = attention_energies.to(encoder_state.device)
        sparsity_transform = Sparsemax(dim=-1)
        all_attentions = []

        for i in range(output_sequence_size):
            # Schedule sampling is whe
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                #print("Feeding the ground truth")
                y_t_index = target_sequence[i]
            else:
                pass
                #print("Not feeding the ground truth")
                
            # Step 1: Embed word and concat with previous context as implemented in Bahdanau et al.
            y_input_vector = self.target_embedding(y_t_index)
            
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)
            
            # Step 2: Make a GRU step, getting a new hidden vector
            h_t = self.gru_cell(rnn_input, h_t)
            
            self._cached_ht.append(h_t.cpu().detach().numpy())
            
            # Step 3: Use the current hidden to attend to the encoder state
            context_vectors, p_attn = self.attention_mechanism(encoder_state_vectors=encoder_state, query_vector=h_t)
            attention_energies = attention_energies + p_attn
            all_attentions.append(p_attn.view(p_attn.size()[0], p_attn.size()[1], 1))
            
            # auxillary: cache the attention probabilities for visualization
            self._cached_p_attn.append(p_attn.cpu().detach().numpy())
            
            ## Step 4: Use the current hidden and context vectors to make a prediction for the next word if following Luong et al.
            # prediction_vector = torch.cat((context_vectors, h_t), dim=1) # denoted by h_t tilda in the paper

            ## Step 4: The only the current hidden state to make a prediction for the next word if following Bahdanau et al.
            prediction_vector = h_t

            ## Linear classifier on top of the prediction vector - If factorization method is not utilized
            # score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.3, training=self.training_mode))
            
            ## Linear classifier on top of the prediction vector - If factorization method is not utilized
            embedding_prediction = self.classifier(F.dropout(prediction_vector, 0.3, training=self.training_mode))

            # Get the indeces of all words included in the target vocab
            # all_target_vocab_indices = []
            # for i in range(0, batch_size):
            #     all_target_vocab_indices.append(list(range(self.num_embeddings)))

            all_target_vocab_indices = torch.arange(0, self.num_embeddings, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(encoder_state.device)

            # Convert the indices to a torch tensor
            #all_target_vocab_indices = torch.LongTensor(all_target_vocab_indices).to(encoder_state.device)

            # This should effectively return a tensor of shape BSxTarget_Vocab_SizexEmb_Dim
            target_vocab_embeddings = self.target_embedding(all_target_vocab_indices)

            # Add an artificial dimension to the embedding predictions
            embedding_prediction = embedding_prediction.unsqueeze(1)

            # Repeat the embeddings prediction along the first axis, so they are suitable for cosine similarity measures
            embedding_prediction = embedding_prediction.repeat(1, self.num_embeddings, 1)

            # Calculate the cosine similarity
            cosine_similarity = torch.nn.functional.cosine_similarity(target_vocab_embeddings,embedding_prediction, dim=2)

            if use_sample:
                #p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                p_y_t_index = F.softmax(cosine_similarity, dim=1)

                if self.training_mode:
                    #print("It is in training mode")
                    # In training mode sample from a multinomial distribution
                    y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()
                
                else:
                    #print("It is not in training mode")
                    y_t_index = torch.argmax(p_y_t_index, 1)

            # print("#######################################")
            # auxillary: collect the prediction scores
            output_vectors.append(cosine_similarity)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        stacked_attentions = torch.stack(all_attentions, dim=2)
        stacked_attentions = stacked_attentions.view(stacked_attentions.size()[0], stacked_attentions.size()[1], stacked_attentions.size()[2])


        return output_vectors, stacked_attentions