import numpy as np

def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    
    # walk over each generated word
    for row in data:
        
        all_candidates = list()
        
        # expand each current candidate
        for i in range(len(sequences)):
            
            # All sequences preserved by the search until now
            seq, score = sequences[i]
            
            # Iterate over all possible options for this word
            for j in range(len(row)):
                
                # Create a new candidate by multiplying the previous score and the new score
                candidate = [seq + [j], score * row[j]]
                
                # Append the new candidate to the list of all candidates
                all_candidates.append(candidate)
                
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)

        # select k best candidates
        sequences = ordered[:k]
        
    return sequences

def sentence_from_indices(indices, vocab, strict=True, return_string=True):
    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []

    if len(indices.shape) == 2:
        indices = np.argmax(indices, axis=1)

    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index))
    if return_string:
        return " ".join(out)
    else:
        return out

def sentence_from_tensor_indices(indices, vocab, strict=True, return_string=True):
    ignore_indices = set([vocab.mask_index, vocab.begin_seq_index, vocab.end_seq_index])
    out = []
    for index in indices:
        if index == vocab.begin_seq_index and strict:
            continue
        elif index == vocab.end_seq_index and strict:
            break
        else:
            out.append(vocab.lookup_index(index.item()))
    if return_string:
        return " ".join(out)
    else:
        return out
