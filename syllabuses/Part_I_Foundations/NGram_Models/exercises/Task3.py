from nltk.lm import NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from collections import defaultdict
from itertools import permutations
from task1 import prepare_corupus, TextFile_Corpus, filter_invalid_ngrams
from nltk.lm import NgramCounter
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math

def addK_smoothing(ng_counter, max_order, k):
    """
    Calculates the Add-k smoothed probabilities for n-grams up to max_order.

    Args:
        ng_counter: An NgramCounter object containing n-gram counts.
        max_order: The maximum n-gram size to calculate probabilities for.
        k: The smoothing parameter (e.g., k=1 for Laplace smoothing).

    Returns:
        A list of dictionaries. The i-th dictionary contains the smoothed
        probabilities for n-grams of size i+1.
    """
    # Vocabulary size is needed for the denominator in the smoothing formula
    # We get it from the unigram counts.
    vocab_size = len(ng_counter[1].keys())
    
    probabilities_list = []
    
    # Iterate through each n-gram order from 1 to max_order
    for order in range(1, max_order + 1):
        prob_dict = {}
        
        if order == 1:
            unigram_counts = ng_counter[1]
            total_unigrams = unigram_counts.N()
            
            
            if total_unigrams > 0:
                for unigram, count in unigram_counts.items():
                    prob = (count ) / total_unigrams
                    prob_dict[unigram] = prob
        else:

            ngram_counts = ng_counter[order]
            context_counts = ng_counter[order - 1]

            all_contexts = list(context_counts.keys())
            all_words = list(ng_counter[1].keys())
            
            for context in all_contexts:
                context_count = context_counts.get(context, 0)
                denominator = context_count + k * vocab_size
                

                for word in all_words:
                    # Get the count of the full n-gram (context, word)
                    # The .get() method returns 0 if the n-gram is unseen.
                    ngram_count = ngram_counts[context].get(word, 0)
                    
                    prob = (ngram_count + k) / denominator
                    ngram = (context,) + (word,)
                    prob_dict[ngram] = prob
        
        probabilities_list.append(prob_dict)
        
    return probabilities_list

corpus = TextFile_Corpus(path="./corpus.txt")
uninq_n_grams, n_grams_flatten, n_grams, train_data, vocab, all_words = prepare_corupus(corpus, n=2)


train_data_list = [list(s) for s in train_data]

ng_counter = NgramCounter(train_data_list)

probabilities = addK_smoothing(ng_counter, 2, 1)


print(probabilities)