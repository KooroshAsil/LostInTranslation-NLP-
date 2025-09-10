# requires: pip install nltk
from nltk.lm import MLE               # or Laplace, KneserNeyInterpolated, ...
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize

# your corpus
corpus = [
    "I am Sam",
    "Sam I am",
    "Sam I like",
    "I do like Sam",
    "do I like Sam",
]

# simple tokenization + lowercase
tokenized = [[w.lower() for w in word_tokenize(sent)] for sent in corpus]

n = 2 
train_data, vocab = padded_everygram_pipeline(n, tokenized)

# choose model: MLE (no smoothing) -- swap to Laplace(n) or KneserNeyInterpolated(n) to add smoothing
model = MLE(order=n)
model.fit(train_data, vocab)

# 1) conditional probability P(token | context)
# NLTK's API: model.score(word, context_list)

p_like_given_i = model.score('like', ['i'])    # P(like | i)
print("P(like | i) =", p_like_given_i)


# 2) next-word distribution (PDF/PMF) given a context
# iterate over vocabulary and score each token (then normalize if you want)
vocab_tokens = list(model.vocab)  # vocabulary object; contains tokens and <UNK>

print(list(vocab_tokens)) # this has a placeholder 'UNK' for unseen words 
raw = {w: model.score(w, ['i']) for w in vocab_tokens}
total = sum(raw.values())
dist = {w: (cnt/total if total>0 else 0.0) for w, cnt in raw.items()}
print("Next-word distribution given context ('i',):")
print({k: v for k, v in dist.items() if v>0})

# 3) sentence probability (product of conditional probs)
def sentence_prob(sentence, model, n):
    toks = [w.lower() for w in word_tokenize(sentence)]
    padded = ['<s>']*(n-1) + toks + ['</s>']
    prob = 1.0
    for i in range(n-1, len(padded)):
        context = padded[i-(n-1):i] if n>1 else []
        w = padded[i]
        p = model.score(w, context)
        prob *= p
        if prob == 0.0:
            return 0.0
    return prob

print("P('I like Sam') =", sentence_prob("I like Sam", model, n))

# 4) perplexity / entropy
# prepare test ngrams (same preprocessing) and call model.perplexity(list_of_ngrams)
# N.B. model.perplexity takes an iterable of ngram tuples (see docs); above we compute sentence_prob manually.
