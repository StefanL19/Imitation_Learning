import torch
from torch.nn import functional as F
from sparsemax import Sparsemax

class BahdanauAttention(torch.nn.Module):
    """
        Bahdanau Attention Mechanism
    """
    def __init__(self, key_size, query_size, attention_function):
        """
        Args:
            key_size (int): The size of each one of the output vectors
                in the encoder. Here it should be 2*hidden_size as 
                the encoder is Bidirectional 
            query_size (int): The size of the decoder hidden state. Usually,
                it can be either hidden_size of the encoder - in case only the
                one of the directions was feeded as a decoder hidden state
                or it can be 2*hidden_size if both directions were feeded, it
                also depends on the output size of the hidden_map layer in the decoder
        """

        super(BahdanauAttention, self).__init__()

        print("The query size is: ", query_size)
        print("The key size is: ", key_size)

        self.query_size = query_size

        # The keys are the encoder hidden states, as in the Bahdanau Paper, we will need a weight matrix to project them
        # We take the query_size as the output size, as it is <= key_size
        self.key_layer = torch.nn.Linear(key_size, query_size, bias=False)

        # The query is the previous state of the cell, as in the Bahdanau Paper, we will need a weight matrix to project it
        self.query_layer = torch.nn.Linear(query_size, query_size, bias=False)
        
        # The final weight matrix which is the the leftmost part in the equation in the Bahdanau paper, it projects the similarity to a value
        self.energy_layer = torch.nn.Linear(query_size, 1, bias=False)

        if attention_function == "sparsemax":
            print("Using sparsemax attention")
            self.attention_calculation = Sparsemax(dim=-1)
        else:
            print("Using softmax attention")
            self.attention_calculation = torch.nn.Softmax(dim=-1)

    def forward(self, encoder_state_vectors, query_vector):
        # mock dirichlet at the beginning
        batch_size, num_vectors, vector_size = encoder_state_vectors.size()


        # Project the query vector
        query = self.query_layer(query_vector)
        query = query_vector.view(batch_size, 1, self.query_size)

        # Project the key vectors 
        # Example Projection: (BS, SEQ_SIZE, 2*HIDDEN_SIZE) - > (BS, SEQ_SIZE, QUERY_SIZE)
        # Although the QUERY_SIZE can be <= 2*HIDDEN_SIZE = KEY_SIZE
        key = self.key_layer(encoder_state_vectors)

        # After the above two operations, query and key should have the same shape along first and last dimension
        # We can sum them, apply tanh and calculate the alphas as in the Bahdanau Paper
        # SCORES_SHAPE = (BS, SEQ_SIZE, 1)
        scores = self.energy_layer(torch.tanh(query+key))

        # CONVERT SCORES_SHAPE TO (BS, SEQ_SIZE)
        scores = scores.squeeze(2).unsqueeze(1)

        # Compute the vector probabilities
        vector_probabilities = self.attention_calculation(scores) #F.softmax(scores, dim=-1)
        
        vector_probabilities = vector_probabilities.view(batch_size, num_vectors)

        # Calculate the context vectors
        weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)

        context_vectors = torch.sum(weighted_vectors, dim=1)

        return context_vectors, vector_probabilities

class Multiplicative_Attention(torch.nn.Module):
    """
        Simple attention mechanism based on dot product vector similarity 
    """
    def __init__(self):
        pass

    def forward(self, encoder_state_vectors, query_vector):
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

        return context_vectors, vector_probabilities

