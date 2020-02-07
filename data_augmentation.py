from slot_aligner import SlotAligner
from nltk.tokenize import word_tokenize
from alignment_utils import tokenize_mr
import nltk

def insert_dot(utt):
    """
        Makes sure that the reference sentence ends with a dot, this is important for the splitting part
    """
    if utt[-1] == ",":
        utt = utt[:-1]+"."
    elif utt[-1] == "!":
        utt = utt[:-1]+"."
    elif utt[-1] == " ":
        utt = utt[:-1]+"."
    elif utt[-1] in "0123456789abcdefghijklmnopqrstuvwxyzéABCDEFGHIJKLMNOPQRSTUVWXYZ'":
        utt += "."
    
    return utt

def define_boundaries(splits):
    """
        Defines the boundaries of each of the sentences based on a split
    """

    sent_boundaries = []
    for idx in range(0, len(splits)-1):
        initial_offset = 1

        if idx == 0:
            initial_offset = 0

        sent_boundaries.append([splits[idx]+initial_offset, splits[idx+1]+1])

    return sent_boundaries



def get_sent_boundaries(utt):
    """
        Gets the position of the last dot sign in each sentence
    """

    sent_text= nltk.sent_tokenize(utt)

    bounds = []
    for idx, sent in enumerate(sent_text):
        if idx == 0:
            bounds.append(len(sent)-1)
        else:
            last_pos = bounds[-1]
            bounds.append(len(sent_text[idx-1])+len(sent))

    new_utt = " ".join(sent_text)
    return bounds, new_utt


def split_utterance(input_mrs, ref):
    aligner = SlotAligner()

    #Make sure that the sentence ends with a dot
    ref = insert_dot(ref)

    # Get boundaries of all sentences inside the utterance
    splits, ref = get_sent_boundaries(ref)

    # If only one sentence - there is nothing to split - just return the utterance
    if len(splits) == 1:
        return [(input_mrs, ref)]

    else: 
        splits = [0] + splits

        # We have as many boundaries as sentences within the utterance
        sent_boundaries = define_boundaries(splits)

        # Get the slots that each one of the sentences realizes
        realizations = [[] for x in range(0,len(sent_boundaries))]

        slot_realization_positions = aligner.align_slots(input_mrs, ref)

        slot_keys = list(input_mrs.keys())

        # Create a list containing the new samples
        # The first new sample is the original 
        new_samples = [(input_mrs, ref)]

        # Iterate over the sentences
        for idx_bound, bound in enumerate(sent_boundaries):
            # Define the new reference sentence according to the boundaries
            new_ref = ref[bound[0]:bound[1]]
            new_ref = new_ref.strip()

            new_mrs = {}
            for idx, pos in enumerate(slot_realization_positions):
                if pos >= bound[0] and pos < bound[1]:
                    slot_key = slot_keys[idx]
                    new_mrs[slot_key] = input_mrs[slot_key]

            # Check if some samples were realized
            if bool(new_mrs):
                # Add inner / outter slot
                if idx_bound == 0:
                    new_mrs["position"] = "outer"
                else:
                    new_mrs["position"] = "inner"

                # Check if the name token was realized
                if not "name" in new_mrs:
                    
                    # If the name slot was not realized directly but there is a pronoun referring to it, then add it
                    if ("it" in new_ref) or ("its" in new_ref):
                        if "name" in slot_keys:
                            new_mrs["name"] = input_mrs["name"]

                new_samples.append((new_mrs, new_ref))

        return new_samples




