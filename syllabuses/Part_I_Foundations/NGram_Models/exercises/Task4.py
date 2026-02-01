import math
from nltk.lm import Lidstone
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from itertools import chain
from task1 import prepare_corupus, TextFile_Corpus, filter_invalid_ngrams


# 2. Prepare the corpus and train the Lidstone model
n = 2  # Bigram model
gamma = 1  # Lidstone smoothing parameter (0 < gamma <= 1)
# Note: Lidstone(gamma=1) is equivalent to Laplace (Add-1) smoothing.

corpus = TextFile_Corpus(path="./corpus.txt")

tokenized_sentences = [[w.lower() for w in word_tokenize(sent)] for sent in corpus]


train_data, vocab = padded_everygram_pipeline(n, tokenized_sentences)

# Instantiate the Lidstone model
lm = Lidstone(gamma, n)

# Fit the model with the prepared data
lm.fit(train_data, vocab)

# 3. Prepare the test sentences for perplexity calculation
test_sentence = "I like Sam"
test_sentence_oov = "I do like Bob"

test_sentences = [test_sentence, test_sentence_oov]



tokenized_test_sentences = [[w.lower() for w in word_tokenize(sent)] for sent in test_sentences]
padded_test_data, _ = padded_everygram_pipeline(n, tokenized_test_sentences)
test_data = [list(g) for g in list(padded_test_data)]
test_data = filter_invalid_ngrams(test_data)


# 4. Use the built-in .perplexity() method
perplexity_score = lm.perplexity(list(test_data))

print(perplexity_score)
