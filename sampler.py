import numpy as np
from utils_inference import sentence_from_indices, beam_search_decoder, sentence_from_tensor_indices
from alignment_utils import tokenize_mr, tokenize_mr_upper
from data_processing import Delexicalizer
from slot_aligner import SlotAligner
import torch
from nltk.translate import bleu_score
from beam_search_allennlp import BeamSearch  
from sparsemax import Sparsemax

class NMTSampler:
    def __init__(self, vectorizer, model, use_reranker, beam_width=3):
        self.vectorizer = vectorizer
        self.model = model
        self.delexicalizer = Delexicalizer("partial", ["name", "near", "food"])
        self.aligner = SlotAligner()
        self.use_reranker = use_reranker
        self.beam_width = beam_width
        self.smoothing_function = bleu_score.SmoothingFunction()
        self._beam_search = BeamSearch(
            vectorizer.target_vocab.end_seq_index, max_steps=45, beam_size=self.beam_width
        )
        self.simplex_projection = Sparsemax(dim=-1)

    def get_prediction_slice(self, idx):
        preds_slice = self._last_batch["unnormalized_predictions"].select(1, idx)
        return preds_slice

    def take_step(self, last_predictions, idx):
        desired_shape = last_predictions.shape[0]

        preds = self.get_prediction_slice(idx)

        class_log_probabilities = self.simplex_projection(preds) #torch.nn.functional.softmax(preds, dim=1)

        idx = idx+1
        repetitions = int(desired_shape/class_log_probabilities.shape[0])
        new_log_probs = class_log_probabilities.repeat(repetitions ,1)

        return new_log_probs, idx

    def apply_to_batch(self, batch_dict):
        self._last_batch = batch_dict

        print(batch_dict["x_source"])

        y_pred, _ = self.model(x_source=batch_dict['x_source'], 
                            x_source_lengths=batch_dict['x_source_length'], 
                            target_sequence=batch_dict['x_target'],
                            sample_probability=1.)

        self._last_batch["unnormalized_predictions"] = y_pred

        # print("The predictions for the two samples are the same: ")
        # print(torch.allclose(y_pred[0], y_pred[1]))

        start_predictions = self._last_batch["unnormalized_predictions"].select(1, 0).unsqueeze(1)
        idx = torch.tensor(0)

        all_top_k_predictions, log_probabilities = self._beam_search.search(
            start_predictions, idx, self.take_step
        )

        # print("The beam search results are the same: ")
        # print(torch.allclose(all_top_k_predictions[0], all_top_k_predictions[1]))
        all_top_k_predictions = all_top_k_predictions.cpu().detach().numpy()
        log_probabilities = log_probabilities.cpu().detach().numpy()
        #print(all_top_k_predictions[0])
        # print(self._last_batch['inp_gt'][0])
        # print(sentence_from_indices(all_top_k_predictions[0][3], self.vectorizer.target_vocab, return_string=True))
        # print("Top k predictions shape: ", all_top_k_predictions.shape)
        # Apply beam search on the predictions

        self._last_batch['y_pred'] = y_pred
        self._last_batch["beam_search_results"] = (all_top_k_predictions, log_probabilities)
        # print(self.model.decoder._cached_p_attn.shape)
        # attention_batched = np.stack(self.model.decoder._cached_p_attn).transpose(1, 0, 2)
        # self._last_batch['attention'] = attention_batched
        
    def _get_source_sentence(self, index, return_string=True):
        indices = self._last_batch['x_source'][index].cpu().detach().numpy()
        vocab = self.vectorizer.source_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_reference_sentence(self, index, return_string=True):
        indices = self._last_batch['y_target'][index].cpu().detach().numpy()
        vocab = self.vectorizer.target_vocab
        return sentence_from_indices(indices, vocab, return_string=return_string)
    
    def _get_gt_mrs(self, index):
        inp_mr = self._last_batch['inp_gt'][index]
        return inp_mr
    
    def _get_gt_ref(self, index):
        ref_gt = self._last_batch['ref_gt'][index]
        return ref_gt
    
    def _get_beam_seach_results(self, index):
        res_seq = self._last_batch["beam_search_results"][0]
        res_probs = self._last_batch["beam_search_results"][1]

        return (res_seq[index], res_probs[index])

    def _get_sampled_sentence(self, index, return_string=True):
        vocab = self.vectorizer.target_vocab

        # Convert the scores for the words to softmax probabilities
        prob_res = self.simplex_projection(self._last_batch['y_pred'])

        sentece_probs = prob_res[index].cpu().detach().numpy()
        print(sentece_probs[5][:200])

        bs_res = self._get_beam_seach_results(index)

        #top_sequences = beam_search_decoder(sentece_probs, self.beam_width)

        gt_mr = tokenize_mr(self._last_batch['inp_gt'][index])

        # By default without reranking the index of the max sequence will be 0
        max_seq_idx = 0

        if self.use_reranker:
            with open("data/results/temp_test.txt", "a") as f:
                f.write(self._last_batch['inp_gt'][index])
                f.write("\n")
                f.write("------------------------------------")
                f.write("\n")
                seq_probs = []
                reranker_probs = []
                for idx, sequence in enumerate(bs_res[0]):
                    sent_str = sentence_from_indices(sequence, vocab, return_string=True)
                    reranker_score, overgen, undergen = self.aligner.alignment_reranker(gt_mr, sent_str)
                    f.write(sent_str)
                    f.write("|||")
                    f.write(str(reranker_score))
                    f.write("|||")
                    f.write(str(index))
                    f.write("||||")
                    f.write(str(overgen))
                    f.write("||||")
                    f.write(str(undergen))
                    f.write("\n")
                    seq_probs.append(bs_res[1][idx]*reranker_score)
                    reranker_probs.append([reranker_score, overgen, undergen])
                f.write("\n\n\n")
            max_seq_idx = seq_probs.index(max(seq_probs))

            #print("The index of the maximal sequence is: ", max_seq_idx)


        # if self.use_reranker:
        #     res_after_reranking = []
        #     # Iterate over all produced sequences
        #     for sequence in top_sequences:
        #         print("The shape of the inernal seq is: ", len(sequence[0]))
        #         # Produce the sentence
        #         sent_str = sentence_from_indices(sequence[0], vocab, return_string=True)

        #         # Get the reranker score
        #         reranker_score = self.aligner.alignment_reranker(gt_mr, sent_str)

        #         # Produce the new sentence probability 
        #         res_after_reranking.append(sequence[1]*reranker_score)  
        #         print(sent_str)
        #     print("------------------")
        #         # Get the sentence with max probability of after both beam search and reranking
        #     max_seq_idx = res_after_reranking.index(max(res_after_reranking))
        #     print(max_seq_idx)

        #     if max_seq_idx != 0:
        #         print("Reranker did its job")
        
        #sentence_indices = top_sequences[max_seq_idx][0]
        sentence_indices = bs_res[0][max_seq_idx]

        # print("Making a prediction {}".format(index))
        # print(sentence_from_indices(sentence_indices, vocab, return_string=True))
        sent_result = sentence_from_indices(sentence_indices, vocab, return_string=return_string)

        if reranker_probs[max_seq_idx][0] < len(gt_mr.keys()):
            with open("data/results/aigner_mistakes_test.txt", "a") as f:
                f.write(self._last_batch['inp_gt'][index])
                f.write("\n")
                f.write("------------------------------------")
                f.write("\n")
                f.write(sent_str)
                f.write("|||")
                f.write(str(reranker_probs[max_seq_idx][0]))
                f.write("\n")
                f.write("\n")

        best_score_overgenerated = reranker_probs[max_seq_idx][1]
        best_score_undergenerated = reranker_probs[max_seq_idx][2]

        # print("----------------------------------")
        return sent_result, best_score_overgenerated, best_score_undergenerated

    def get_ith_item(self, index, return_string=True):
        sampled_sentence, best_score_overgenerated, best_score_undergenerated = self._get_sampled_sentence(index, return_string=return_string)

        gt_mrs = self._get_gt_mrs(index)
        ref_gt = self._get_gt_ref(index)
        
        sampled_normalized = self.delexicalizer.reverse_delexicalize_sample(gt_mrs, " ".join(sampled_sentence))
        
        output = {"source": self._get_source_sentence(index, return_string=return_string), 
                  "reference": self._get_reference_sentence(index, return_string=return_string), 
                  "sampled": sampled_sentence,
                  # "attention": self._last_batch['attention'][index],
                  "sampled_normalized":sampled_normalized,
                  "reference_gt":ref_gt,
                  "mrs_gt":gt_mrs,
                  "y_target": self._last_batch['y_target'][index],
                  "overgenerated":best_score_overgenerated,
                  "undergenerated":best_score_undergenerated
                 }
        
        reference = output['reference']
        hypothesis = output['sampled']
        
        if not return_string:
            reference = " ".join(reference)
            hypothesis = " ".join(hypothesis)
        
        output['bleu-4'] = bleu_score.sentence_bleu(references=[reference],
                                                    hypothesis=hypothesis,
                                                    smoothing_function=self.smoothing_function.method1)
        
        return output

        