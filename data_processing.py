import pandas as pd
from tqdm import tqdm
import re
import nltk
from slot_aligner import SlotAligner
from data_augmentation import split_utterance
from alignment_utils import tokenize_mr, tokenize_mr_upper



class DataPreprocessor(object):
    def __init__(self, dataframe=None, delexicalization_type="full", delexicalization_slots=None):
        self.delexicalization_type = delexicalization_type
        self.dataframe = dataframe
        self.delexicalization_slots = delexicalization_slots

    @classmethod
    def from_existing_df(cls, existing_df_path, delexicalization_type="full", delexicalization_slots=None):
        df = pd.read_csv(existing_df_path)
        return cls(df, delexicalization_type, delexicalization_slots)

    @classmethod 
    def from_files(cls, train_input_path, validation_input_path, test_input_path,  delexicalization_type="full", delexicalization_slots=None):
        """
            Instantiates the DataPreprocessor class from a list of training, validation, and test data paths

            Args:
                train_input_path (string): Input path to csv file containing the training samples
                validation_input_path (string): Input path to the csv file containing the validation samples
                test_input_path (string): Input path to the csv file containing the test samples
                delexicalization_type (string): One of: full, partial, none
                delexicalization_slots (array): Slots to be delexicalized, none if delexicalization type is full or none
        """
        train_df = pd.read_csv(train_input_path)
        validation_df = pd.read_csv(validation_input_path)
        test_df = pd.read_csv(test_input_path)

        delexicalizer = Delexicalizer(delexicalization_type, delexicalization_slots)
        aligner = SlotAligner()

        # Some dictionaries to keep track on important statistics
        delexicalization_failures = {
            "name":[],
            "near":[],
            "food":[],
            "area":[]
            }

        # Time to align the slots, to actually check if they are realized in the utterances
        aligner_failures = {
        "name":[],
        "near":[],
        "food":[],
        "eattype":[],
        "pricerange":[],
        "customerrating":[],
        "area":[],
        "familyfriendly":[]
        }

        aligner_successes = {
        "name":[],
        "near":[],
        "food":[],
        "eattype":[],
        "pricerange":[],
        "customerrating":[],
        "area":[],
        "familyfriendly":[]
        }

        preprocessed_mrs = []
        preprocessed_ref_sentences = []

        mr_no_preprocessing = []

        input_language = []
        output_language = []
        ref_no_processing = []
        for index, row in tqdm(train_df.iterrows()):

            input_mr = tokenize_mr(row[0])
            
            # The customer rating slot should be renamed - empty spaces in a slot are not a good thing :D 
            if "customer rating" in input_mr.keys():
                input_mr["customerrating"] = input_mr.pop("customer rating")

            # Convert the values of all input_mrs to lower
            for mr in input_mr.keys():
                input_mr[mr] = input_mr[mr].lower()

            output_ref = row[1]
            
            output_ref = output_ref.lower()
            # Remove any repeating whitespaces
            output_ref = ' '.join(output_ref.split())
            
            input_mr, output_ref, delexicalization_results = delexicalizer.delexicalize_sample(input_mr, output_ref)
            
            # Inspect delexicalization results, we do not want to end up with slots that were not delexicalized
            # or at least we want to know about it
            for result in delexicalization_results:
                # If the delexicalization process was not successfull
                if result[1] == False:
                    # Keep track on which samples had failed delexicalization results
                    delexicalization_failures[result[0]].append(index)

            # Time to do some slot allignment to check if the slots are actually realized in the utterances
            alignment_results = aligner.align_slots(input_mr, output_ref)
            
            keys_to_remove = []
            for idx, res in enumerate(alignment_results):
                slot_key = list(input_mr.keys())[idx]

                # If the result is negative, then the slot was not realized
                if res == -1:
                    # Remove the key from the slots if it was not realized
                    keys_to_remove.append(slot_key)

                    # :D Keep some statistics of the non-aligned slots
                    aligner_failures[slot_key].append(index)
                else:
                    # Well, keep some statistics of the aligned slots as well :D
                    aligner_successes[slot_key].append(index)

            for sk in keys_to_remove:
                input_mr.pop(sk)

            # TO DO: this may cause some hallucinations related to the city, it is not a surprise that TUDA augmented set likes to place city
            # in the utterance the word is present really often in the dataset, so the hallucination is kind of expected
            
            # if "city" in output_ref:
            #     print(input_mr)
            #     print(output_ref)
            #     print("$$$$$$$$$$$$$$$$$$$$$$$$$")

            # Well, everything is fine, now it is time for splitting the utterance if possible
            samples = split_utterance(input_mr, output_ref)

            for sample in samples:
                # Construct the input language and the output language samples
                mrs = sample[0]
                ref = sample[1]

                inp = []
                for slot_name in list(mrs.keys()):
                    #inp.append("<inform>")
                    inp.append(slot_name)
                    inp.append(mrs[slot_name])

                # Punctuation should be a separate token 
                ref = re.sub('([.,!?()])', r' \1 ', ref)
                ref = re.sub('\s{2,}', ' ', ref)
                inp_sent = " ".join(inp)
                if "nan" in str(inp_sent).lower():
                    print(inp_sent)
                    #Something went wrong
                    print("The sentence is processed in a wrong way: ")
                    print(row[0])
                    print(row[1])
                    continue

                input_language.append(inp_sent)
                output_language.append(ref)
                mr_no_preprocessing.append(row[0])

                # A dummy string for the training gt without delexicalization - we do not actually need it except for the dataframe
                ref_no_processing.append(" ")

        print("Input training samples: ", len(input_language))
        print("Output training samples: ", len(output_language))
        train_description = ["train"]*len(input_language)


        train_df = pd.DataFrame({'source_language': input_language, 'target_language': output_language,
         'split': train_description, 'inp_gt':mr_no_preprocessing, 'ref_gt':ref_no_processing})
        print("-------------------------------------------------")
        print(train_df)

        # Inspect delexicalization failures for each of the three slots

        print("#######################Delexicalization Failures####################################")
        for slot_name in delexicalization_failures.keys():
            print(slot_name + " : " + str(len(delexicalization_failures[slot_name])))
        
        print("#######################Alignment Successes####################################")
        # Inspect slot alignment failures for each of the slots
        for slot_name in aligner_successes.keys():
            print(slot_name + " : " + str(len(aligner_successes[slot_name])))

        print("#######################Alignment Failures####################################")
        # Inspect slot alignment failures for each of the slots
        for slot_name in aligner_failures.keys():
            print(slot_name + " : " + str(len(aligner_failures[slot_name])))

        val_input_language = []
        val_output_language = []
        val_mr_no_preprocessing = []
        val_ref_no_processing = []

        for index, row in tqdm(validation_df.iterrows()):
            input_mr = tokenize_mr(row[0])

            # The customer rating slot should be renamed - empty spaces in a slot are not a good thing :D 
            if "customer rating" in input_mr.keys():
                input_mr["customerrating"] = input_mr.pop("customer rating")

            # Convert the values of all input_mrs to lower
            for mr in input_mr.keys():
                input_mr[mr] = input_mr[mr].lower()

            output_ref = row[1]
            
            output_ref = output_ref.lower()
            # Remove any repeating whitespaces
            output_ref = ' '.join(output_ref.split())
            
            input_mr, output_ref, delexicalization_results = delexicalizer.delexicalize_sample(input_mr, output_ref)

            inp_val = []
            for slot_name in list(input_mr.keys()):
                    #inp_val.append("<inform>")
                    inp_val.append(slot_name)
                    inp_val.append(input_mr[slot_name])


            # Punctuation should be a separate token 
            output_ref = re.sub('([.,!?()])', r' \1 ', output_ref)
            output_ref = re.sub('\s{2,}', ' ', output_ref)
            ref_gt = row[1]
            ref_gt = re.sub('([.,!?()])', r' \1 ', ref_gt)
            ref_gt = re.sub('\s{2,}', ' ', ref_gt)

            inp_sent = " ".join(inp_val)
            val_input_language.append(inp_sent)
            val_output_language.append(output_ref)
            val_mr_no_preprocessing.append(row[0])
            val_ref_no_processing.append(ref_gt)

        print("Input validation samples: ", len(val_input_language))
        print("Input vaidation samples: ", len(val_output_language))
        print("------------------------------------")

        val_description = ["val"]*len(val_input_language)

        val_df = pd.DataFrame({'source_language': val_input_language, 'target_language': val_output_language,
         'split': val_description, 'inp_gt':val_mr_no_preprocessing, 'ref_gt':val_ref_no_processing})

        df = pd.concat([train_df, val_df], keys=['source_language', 'target_language', 'split', 'inp_gt', 'ref_gt'])

        return cls(df, delexicalization_type, delexicalization_slots)

    def add_samples(self, samples_file_path, original_train_path):
        with open(samples_file_path, "r") as f:
            lines = f.readlines()
        
        delexicalizer = Delexicalizer(self.delexicalization_type, self.delexicalization_slots)

        # Load the training csv, so the mrs can be extracted
        train_df = pd.read_csv(original_train_path)

        input_language = []
        output_language = []
        mr_no_preprocessing = []
        ref_no_processing = []

        print(train_df.iloc[100])
        print("$$$$$$$$$$$$$$$$$")

        for idx, line in tqdm(enumerate(lines[:10000])):
            line = line.strip("\n")
            orig_train_sample = train_df.iloc[idx]
            input_mr = orig_train_sample[0]
            
            input_mr = tokenize_mr(input_mr)
            
            # The customer rating slot should be renamed - empty spaces in a slot are not a good thing :D 
            if "customer rating" in input_mr.keys():
                input_mr["customerrating"] = input_mr.pop("customer rating")

            # Convert the values of all input_mrs to lower
            for mr in input_mr.keys():
                input_mr[mr] = input_mr[mr].lower()

            line = line.lower()

            input_mr, output_ref, delexicalization_results = delexicalizer.delexicalize_sample(input_mr, line)

            inp = []
            for slot_name in list(input_mr.keys()):
                #inp.append("<inform>")
                inp.append(slot_name)
                inp.append(input_mr[slot_name])

            # Punctuation should be a separate token 
            ref = re.sub('([.,!?()])', r' \1 ', output_ref)
            ref = re.sub('\s{2,}', ' ', ref)

            inp_sent = " ".join(inp)
            # print(inp)
            # print("--------------------------------")
            # print(ref)

            input_language.append(inp_sent)
            output_language.append(ref)
            mr_no_preprocessing.append(orig_train_sample[0])

            # Append an empty stringfor the no preprocessed refernce - we do not need it for the training class
            ref_no_processing.append(" ")

        print("Input training samples: ", len(input_language))
        print("Output training samples: ", len(output_language))
        train_description = ["train"]*len(input_language)

        new_df = pd.DataFrame({'source_language': input_language, 'target_language': output_language,
         'split': train_description, 'inp_gt':mr_no_preprocessing, 'ref_gt':ref_no_processing})
        
        self.dataframe = pd.concat([self.dataframe, new_df], keys=['source_language', 'target_language', 'split', 'inp_gt', 'ref_gt'])

            

    def save_data(self, output_path):
        self.dataframe.to_csv(output_path, encoding='utf-8', index=False)

class Delexicalizer(object):
    def __init__(self, delexicalization_type, delexicalization_slots=None):
        self.delexicalization_type = delexicalization_type
        self.delexicalization_slots = delexicalization_slots
        self.aligner = SlotAligner()
        self.area_lookup_values = {
            # many of the samples contain city centre info and a riverside value - change that
            "riverside":["river", "north of the city", "city centre", "centre of the city", "outskirts of the city", "city center"],
            "city centre": ["city center", "centre of the city", "center of the city", "downtown", "center", "middle city", "middle town", "city"]
        }

    def delexicalize_sample(self, inp, output=None):
        """
            Delexicalizes an input sample

            Args:
            inp(Dictionary): Dictionary containing the input slot types and the corresponding values
            output(string): The reference sentence
        """

        if self.delexicalization_type == "full":
            inp, output, delexicalization_results = self._full_delexicalization(inp, output)
        elif self.delexicalization_type == "partial":
            inp, output, delexicalization_results = self._partial_delexicalization(inp, output)
        else:
            pass

        return inp, output, delexicalization_results

    def reverse_delexicalize_sample(self, inp_mr, model_output):
        """
            Reverses the delexicalization process
        """
        if self.delexicalization_slots == None:
            print("Pass slots to the delexicalizer")
            return

        mr = tokenize_mr_upper(inp_mr)

        for slot in self.delexicalization_slots:
            if slot == "name":
                # Get the name field from the dictionary if it exists
                if "name" in mr.keys():
                    name_val = mr["name"]
                    model_output = self._reverse_delexicalize_name(name_val, model_output)

            if slot == "near":
                #Get the near field from the dictionary if it exists
                if "near" in mr.keys():
                    near_val = mr["near"]
                    model_output = self._reverse_delexicalize_near(near_val, model_output)

            if slot == "food":
                #Get the food field from the dictionary of it exists
                if "food" in mr.keys():
                    food_val = mr["food"]
                    model_output = self._reverse_delexicalize_food_slug2slug(food_val, model_output)

            if slot == "area":
                if "area" in mr.keys():
                    area_val = mr["area"]
                    model_output = self._reverse_delexicalize_area(area_val, model_output)

        return model_output


    def _reverse_delexicalize_area(self, inp, model_output):
        model_output = model_output.replace("x-area", inp)
        return model_output

    def _full_delexicalization(self, inp, output):
        """
            Performs full delexicalization of an input sample

            Args:
            inp(Dictionary): Dictionary containing the input slot types and the corresponding values
            output(string): The reference sentence
        """
        pass

    def _partial_delexicalization(self, inp, output):
        """
            Performs partial delexicalization of an input sample

            Args:
            inp(Dictionary): Dictionary containing the input slot types and the corresponding values
            output(string): The reference sentence
        """
        # A list to keep track on successfull/unsuccesfull delexicalizations
        delexicalization_results = []
        if "name" in self.delexicalization_slots:
            inp, output, is_success = self._delexicalize_name(inp, output)
            delexicalization_results.append(("name", is_success))

        if "near" in self.delexicalization_slots:
            inp, output, is_success = self._delexicalize_near(inp, output)
            delexicalization_results.append(("near", is_success))

        if "food" in self.delexicalization_slots:
            inp, output, is_success = self._delexicalize_food_slug2slug(inp, output)
            delexicalization_results.append(("food", is_success))

        if "familyfriendly" in self.delexicalization_slots:
            print("Delexicalizing family friendly")
            inp, output, is_success = self._delexicalize_family_friendly(inp, output)

        if "pricerange" in self.delexicalization_slots:
            print("Delexicalizing pricerange")
            self._delexicalize_price_range(inp, output)

        if "area" in self.delexicalization_slots:
            #print("Delexicalizing area")
            inp, output, is_success = self._delexicalize_area(inp, output)

            delexicalization_results.append(("area", is_success))


        return inp, output, delexicalization_results

    def _delexicalize_name(self, inp, output):
        """
            Delexicalizes the name field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        # Variable to check if the delexicalization was successfull
        # A delexicalization is unsuccessfull iff the name slot is present in 
        # the mr but the value is not found in the output with the used technique
        
        is_success = True
        if "name" in inp.keys():
            name_value = inp["name"]
            inp["name"] = "x-name"
            
            if output != None:
                # We are not at testing time
                # Check if the name can be found in the output
                if name_value in output:
                    output = output.replace(name_value, "x-name")
                else:
                    is_success = False
        
        return inp, output, is_success

    def _reverse_delexicalize_name(self, inp, model_output):
        """
            Reverses the delexicalization of the name field
        """

        model_output = model_output.replace("x-name", inp)
        return model_output

    def _delexicalize_eat_type(self, sample):
        """
            Delexicalizes the name field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        pass

    def _delexicalize_price_range(self, inp, output):
        """
            Delexicalizes the pricerange field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        is_success = True
        if "pricerange" in inp.keys():
            range_value = inp["pricerange"]
            with open("pricerange_realizations.txt", "a") as f:
                if range_value in output:
                    f.write("Detected ||| " + range_value + " |||| " + output + "\n")
                else:
                    f.write("Non detected ||| " + range_value + " |||| " + output + "\n")
        else:
            print("No pricerange in the slots")
            print("......................")

    def _delexicalize_customer_rating(self, sample):
        """
            Delexicalizes the name field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        pass

    def _delexicalize_near(self, inp, output):
        """
            Delexicalizes the name field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        is_success = True
        if "near" in inp.keys():
            near_value = inp["near"]
            inp["near"] = "x-near"
            
            if output != None:
                # We are not at testing time
                # Check if the name can be found in the output
                if near_value in output or "crown plaza hotel" in output:
                    output = output.replace(near_value, "x-near")

                    # Search specifically for Crowne Hotel Plaza as
                    # it is a problem in many of the samples
                    output = output.replace("crown plaza hotel", "x-near")
                else:
                    is_success = False
        
        return inp, output, is_success

    def _reverse_delexicalize_near(self, inp_val, model_output):
        """
            Reverses the delexicalization of the near field
        """
        model_output = model_output.replace("x-near", inp_val)
        return model_output

    def _delexicalize_food_slug2slug(self, inp, output):
        """
            Delexicalizes the food field in an input/output pair
            sample: input, output pair to be delexicalized
        """
        # The food delexicalization idea was taken from the slug2slug model as the name of the function states
        # https://www.macs.hw.ac.uk/InteractionLab/E2E/final_papers/E2E-Slug2Slug.pdf

        is_success = False
        if "food" in inp.keys():
            food_value = inp["food"]

            # Different cuisine names starting with a vowel
            cuisine_vow = ["italian", "english", "indian"]

            # Different cuisine names starting with a consonant
            cuisine_con = ["japanese", "french", "chinese"]

            # Different types of food 
            food = ["fast food"]

            if food_value in cuisine_vow:
                inp["food"] = "x-vow-cuisine-food"

            elif food_value in cuisine_con:
                inp["food"] = "x-con-cuisine-food"

            else:
                inp["food"] = "x-con-food"

            if output != None:
                for cv in cuisine_vow:
                    if cv in output:
                        is_success = True
                        output = output.replace(cv, "x-vow-cuisine-food")

                for cc in cuisine_con:
                    if cc in output:
                        is_success = True
                        output = output.replace(cc, "x-con-cuisine-food")

                for f in food:
                    if f in output:
                        is_success = True
                        output = output.replace(f, "x-con-food")
        else:
            is_success = True

        return inp, output, is_success

    def _reverse_delexicalize_food_slug2slug(self, inp, model_output):
        """
            Reverses the delexicalization of the food field 
        """
        #Try to consecutively replace all the possible placeholders 
        
        model_output = model_output.replace("x-vow-cuisine-food", inp)

        model_output = model_output.replace("x-con-cuisine-food", inp)

        model_output = model_output.replace("x-con-food", inp)

        return model_output

    def _delexicalize_area(self, inp, output):
        """
            Delexicalizes the name field in an input/output pair
            sample: input, output pair to be delexicalized
        """

        is_success = False

        if "area" in inp.keys():
            area_value = inp["area"]

            if area_value in output:
                is_success = True
                output = output.replace(area_value, "x-area")
                inp["area"] = "x-area"

            else:
                ### Try the alternative value
                alternative_values = self.area_lookup_values[area_value]

                for value in alternative_values:
                    if value in output:
                        output = output.replace(value, "x-area")
                        inp["area"] = "x-area"
                        is_success = True
                        break

            # Note that this also makes sense, but may be removed
            # if "city centre" in output:
            #             print(output)
            #             print("###########################")

            # This slot contains as much noise as actual data -> remove the noise
            if is_success == False:
                del inp["area"]

        else:
            # Specifically for the area slot - there are many places where the slot is hallucinated (bot present in the input slots, but present in the output)
            all_values = self.area_lookup_values["city centre"] + self.area_lookup_values["riverside"] + ["riverside"] + ["city centre"]

            for val in all_values:
                if val in output:
                    inp["area"] = "x-area"
                    output = output.replace(val, "x-area")

            is_success == True

        return inp, output, is_success

    def _delexicalize_family_friendly(self, inp, output):
        """
            Delexicalizes the family-friendly field in an input/output pair
            sample: input, output pair to be delexicalized
            TODO: Figure out delexicalization for this slot
        """

        if "familyfriendly" in inp.keys():
            family_friendly_value = inp["familyfriendly"]
            res = self.aligner.align_familyfriendly(output, family_friendly_value)
            print("####", family_friendly_value)
            print(output[res:])
            if family_friendly_value == "no":
                print(output)
        else:
            print("family friendly not found in the keys")

        return inp, output, True
