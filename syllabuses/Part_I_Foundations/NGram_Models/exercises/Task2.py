"""
Task2 
get probabilities 

"""
from task1 import prepare_corupus
from task1 import TextFile_Corpus
from nltk.lm import NgramCounter
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk import FreqDist
    
def get_all_ngram_probabilities(model, all_ngrams_flattened):
    """
    Uses a trained MLE model to get probabilities for all n-grams.

    Args:
        model: The trained MLE language model.
        all_ngrams_flattened: A dictionary of flattened lists of n-grams,
                              keyed by their order (e.g., '1-grams', '2-grams').

    Returns:
        A list of dictionaries containing all n-gram probabilities for each order.
    """
    probabilities_list = [[] for _ in range(model.order)]
    for k in range(1, model.order + 1):
        prob_dict = {}
        ngram_list = all_ngrams_flattened[f'{k}-grams']
        for ngram in ngram_list:
            if k == 1:
                # Unigrams have no context
                prob = model.score(ngram[0])
            else:
                # Higher-order n-grams have a context
                context = ngram[:-1]
                word = ngram[-1]
                prob = model.score(word, context)
            prob_dict[ngram] = prob
        
        probabilities_list[k-1].append(prob_dict)
    return probabilities_list


corpus = TextFile_Corpus(path="./corpus.txt")
uninq_n_grams, n_grams_flatten, n_grams, train_data, vocab, all_words = prepare_corupus(corpus, n=2)

train_data_list = [list(s) for s in train_data]


ng_counter = NgramCounter(train_data_list)


# 4. Calculate probabilities using the new function
n = 2 # Let's calculate for unigrams, bigrams, and trigrams
lm = MLE(n)
lm.fit(train_data, vocab)

probabilities = get_all_ngram_probabilities(lm, n_grams_flatten)
print(probabilities)