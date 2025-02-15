import boolean_slot
import categorical_slots
import scalar_slots
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

class SlotAligner(object):
    def __init__(self):
        pass

    def alignment_reranker(self, gt_slots, generated_sentence):
        """
            Generates a rank based on aligned, unaligned and overgenerated slots

            gt_slots: a dictionary containing the ground truth slots and their values
            generated_sentence: the sentence that was generated by the model
        """

        # substitute the exact matches with their values
        # Construct a dictionary with all slots
        all_slots = {
        "name":"x-name",
        "near":"x-near",
        "eattype":"-",
        "pricerange":"-",
        "customerrating":"-",
        "food":"-",
        "area":"-",
        "familyfriendly":"-"
        }

        if "customer rating" in gt_slots.keys():
            gt_slots["customerrating"] = gt_slots.pop("customer rating")

        # Iterate the ground truth slots for the current sentence
        for gt_slot_name in gt_slots.keys():
            if gt_slot_name not in ["name", "near"]:  
                # Assign them to the values of the created dictionary
                all_slots[gt_slot_name] = gt_slots[gt_slot_name]

        aligned_slots = self.align_slots(all_slots, generated_sentence, reranking=True)

        # The food slot will be specific because of its delexicalization
        food_values = ["x-vow-cuisine-food", "x-con-cuisine-food", "x-con-food"]
        
        if any(x in generated_sentence for x in food_values):
            # Mark the food slot as realized
            aligned_slots[-3] = 0

        count_all = len(list(gt_slots.keys()))

        # Slots that were realized but not present in the ground truth mr
        count_overgenerated = 0
        
        # Slots that were not realized in the sentence
        count_undergenerated = 0    
        
        for idx, slot in enumerate(all_slots.keys()):
            realization_value = aligned_slots[idx]

            if (slot in gt_slots.keys()) and (realization_value == -1):
                count_undergenerated += 1

            elif (slot not in gt_slots.keys()) and (realization_value >= 0):
                count_overgenerated += 1

        alignment_result = count_all / ((count_overgenerated+1)*(count_undergenerated+1))
        
        return alignment_result, count_overgenerated, count_undergenerated




    def align_slots(self, mr_slots, ref_sentence, reranking=False):
        slots_realization_positions = []

        for slot in mr_slots.keys():
            if slot == "name":
                pos = self.align_name(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)
    
            elif slot == "eattype":
                pos = self.align_eattype(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)

            elif slot == "pricerange":
                pos = self.align_pricerange(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)

            elif slot == "customerrating":
                pos = self.align_customer_rating(ref_sentence, mr_slots[slot], reranking)
                slots_realization_positions.append(pos)

            elif slot == "near":
                pos = self.align_near(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)

            elif slot == "food":
                pos = self.align_food(ref_sentence, mr_slots[slot])
                if reranking == True:
                    pos = -1
                slots_realization_positions.append(pos)

            elif slot == "area":
                pos = self.align_area(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)

            elif slot == "familyfriendly":
                pos = self.align_familyfriendly(ref_sentence, mr_slots[slot])
                slots_realization_positions.append(pos)

        return slots_realization_positions

    def align_name(self, ref_text, value):
        # We are looking for an exact match here
        pos, slot_cnt = self.align_perfect_match(ref_text, value)
        return pos#, slot_cnt

    def align_near(self, ref_text, value):
        # We are looking for an exact match here
        pos, slot_cnt = self.align_perfect_match(ref_text, value)
        return pos#, slot_cnt


    def align_eattype(self, ref_text, value):
        ref_text_tokenized = word_tokenize(ref_text)

        #Using the soft aligner 
        pos = categorical_slots.align_categorical_slot(ref_text, ref_text_tokenized, "eattype", value, mode='first_word')
        return pos

    def align_pricerange(self, ref_text, value):
        ref_text_tokenized = word_tokenize(ref_text)

        #Using the soft aligner
        pos = scalar_slots.align_scalar_slot(ref_text, ref_text_tokenized, "pricerange", value, slot_stem_only=True)
        return pos

    def align_customer_rating(self, ref_text, value, reranking_mode=False):
        ref_text_tokenized = word_tokenize(ref_text)

        customerrating_mapping = {
                                    'slot': 'rating',
                                    'values': {
                                        'low': 'poor',
                                        'average': 'average',
                                        'high': 'excellent',
                                        '1 out of 5': 'poor',
                                        '3 out of 5': 'average',
                                        '5 out of 5': 'excellent'
                                    }
                                }
        #Using the soft aligner
        pos = scalar_slots.align_scalar_slot(ref_text, ref_text_tokenized, "customerrating", value,
                                            slot_mapping=customerrating_mapping['slot'],
                                            value_mapping=customerrating_mapping['values'],
                                            slot_stem_only=True,
                                            reranking=reranking_mode)
        return pos

    def align_area(self, ref_text, value):
        ref_text_tokenized = word_tokenize(ref_text)

        # Using the soft aligner - first_word
        pos = categorical_slots.align_categorical_slot(ref_text, ref_text_tokenized, "area", value, mode='first_word')
        return pos

    def align_familyfriendly(self, ref_text, value):
        ref_text_tokenized = word_tokenize(ref_text)
        
        pos = boolean_slot.align_boolean_slot(ref_text, ref_text_tokenized, "familyfriendly", value)

        return pos

    def align_perfect_match(self, ref_text, value):
        pos = ref_text.find(value)

        slot_cnt = ref_text.count(value)

        return  pos, slot_cnt

    def align_food(self, text, value):
        value = value.lower()

        # This conveniently solves the problem with the delexicalized slots
        pos = text.find(value)

        if pos >= 0:
            return pos

        elif value == 'english':
            return text.find('british')
        elif value == 'fast food':
            return text.find('american style')
        else:
            text_tok = word_tokenize(text)
            for token in text_tok:
                #print(token)

                # FIXME warning this will be slow on start up
                synsets = wordnet.synsets(token, pos='n')
                synset_ctr = 0

                for synset in synsets:
                    synset_ctr += 1
                    hypernyms = synset.hypernyms()

                    # If none of the first 3 meanings of the word has "food" as hypernym, then we do not want to
                    #   identify the word as food-related (e.g. "center" has its 14th meaning associated with "food",
                    #   or "green" has its 7th meaning accociated with "food").
                    while synset_ctr <= 3 and len(hypernyms) > 0:
                        lemmas = [l.name() for l in hypernyms[0].lemmas()]
                        #print(lemmas)
                        if 'chemical-compound' in lemmas:
                            break

                        elif 'beverage' in lemmas:
                            break

                        elif 'food' in lemmas:
                            # print(lemmas)
                            # print(token)
                            return text.find(token)
                        

                        # Follow the hypernyms recursively up to the root
                        hypernyms = hypernyms[0].hypernyms()
                #print("---------------------------")

        return pos 