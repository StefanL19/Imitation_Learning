import pandas as pd
from nltk.util import ngrams 
from collections import Counter

class NGramExtractor(object):
	def __init__(self, dataframe=None, top_n=10):
		self.df = pd.read_csv("data/inp_and_gt_name_near_food_no_inform.csv")
		text = ""

		utterances = self.df["target_language"].to_list()

		for utt in utterances:
			text += utt

		self.all_bigrams = Counter(ngrams(text.split(), 2))
		self.all_trigrams = Counter(ngrams(text.split(), 3))
		self.top_n = top_n

	def islice(self, iterable, *args):
	    # islice('ABCDEFG', 2) --> A B
	    # islice('ABCDEFG', 2, 4) --> C D
	    # islice('ABCDEFG', 2, None) --> C D E F G
	    # islice('ABCDEFG', 0, None, 2) --> A C E G
	    s = slice(*args)
	    start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
	    it = iter(range(start, stop, step))
	    try:
	        nexti = next(it)
	    except StopIteration:
	        # Consume *iterable* up to the *start* position.
	        for i, element in zip(range(start), iterable):
	            pass
	        return
	    try:
	        for i, element in enumerate(iterable):
	            if i == nexti:
	                yield element
	                nexti = next(it)
	    except StopIteration:
	        # Consume to *stop*.
	        for i, element in zip(range(i + 1, stop), iterable):
	            pass

	def take(self, n, iterable):
	    "Return first n items of the iterable as a list"
	    return list(self.islice(iterable, n))

	def get_top_bigrams(self, head_word):
		top_bigrams = {k: v for k, v in self.all_bigrams.items() if k[0]==head_word}

		top_bigrams = self.take(self.top_n, top_bigrams.items())

		bigrams_to_return = []
		for bigram in top_bigrams:
			bigram_txt = bigram[0][0]+" "+bigram[0][1]
			bigrams_to_return.append(bigram_txt)

		return bigrams_to_return

	def get_top_trigrams(self, head_word):
		top_trigrams = {k: v for k, v in self.all_trigrams.items() if k[0]==head_word}

		top_trigrams = self.take(self.top_n, top_trigrams.items())
		trigrams_to_return = []
		for trigram in top_trigrams:
			trigram_txt = trigram[0][0]+" "+trigram[0][1]
			trigrams_to_return.append(trigram_txt)

		return trigrams_to_return



# extractor = NGramExtractor()
# print(extractor.get_top_trigrams("x-vow-cuisine-food"))

