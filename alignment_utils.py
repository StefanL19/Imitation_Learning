import json

def find_first_in_list(val, lst):
    idx = -1
    pos = -1

    for i, elem in enumerate(lst):
        if val == elem:
            idx = i

    if idx >= 0:
        # Calculate approximate character position of the matched value
        punct_cnt = lst[:idx].count('.') + lst[:idx].count(',')
        pos = len(' '.join(lst[:idx])) + 1 - punct_cnt

    return idx, pos


def find_all_in_list(val, lst):
    indexes = []
    positions = []

    for i, elem in enumerate(lst):
        if val == elem:
            indexes.append(i)

            # Calculate approximate character position of the matched value
            punct_cnt = lst[:i].count('.') + lst[:i].count(',')
            positions.append(len(' '.join(lst[:i])) + 1 - punct_cnt)

    return indexes, positions


def get_slot_value_alternatives(slot):
    with open("data/alternatives.json", 'r') as f_alternatives:
        alternatives_dict = json.load(f_alternatives)

    return alternatives_dict.get(slot, {})

def tokenize_mr(sample):
        """
            Performs initial tokenization of the input sample
            sample: input, output pair to be tokenized
            mode: train, validation, or test mode
        """
        output = {}

        mr_parts = sample.split(",")

        for mr_part in mr_parts:
            slot_type = mr_part.split("[")[0].strip().lower()
            slot_value = mr_part.split("[")[1].strip("]").strip().lower()
            
            output[slot_type] = slot_value.lower()

        return output

def tokenize_mr_upper(sample):
        """
            Performs initial tokenization of the input sample
            sample: input, output pair to be tokenized
            mode: train, validation, or test mode
        """
        output = {}

        mr_parts = sample.split(",")

        for mr_part in mr_parts:
            slot_type = mr_part.split("[")[0].strip().lower()
            slot_value = mr_part.split("[")[1].strip("]").strip()
            
            output[slot_type] = slot_value

        return output