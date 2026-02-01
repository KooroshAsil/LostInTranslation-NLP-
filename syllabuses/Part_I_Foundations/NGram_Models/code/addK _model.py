# Install nltk if not installed
# !pip install nltk

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Lidstone
from nltk.tokenize import word_tokenize

# -------------------------
# 1. Your corpus
# -------------------------


with open("./corpus.txt") as f:
    corpus = [w.strip() for w in f.readlines()]

tokenized = [[w.lower() for w in word_tokenize(sent)] for sent in corpus]


n = 2  # bigram model

# -------------------------
# 2. Prepare training data
# -------------------------
train_data, vocab = padded_everygram_pipeline(n, tokenized)

# -------------------------
# 3. Train Lidstone (Add-k) model
# -------------------------
k = 1  # You can try 0.1, 0.5, 1.0 etc.
model = Lidstone(order=n, gamma=k)  # gamma=k does Add-k
model.fit(train_data, vocab)

# -------------------------
# 4. Query probabilities
# -------------------------
context = ["I"]
word = "like"
p = model.score(word, context)
print(f"P({word} | {context[-1]}) = {p:.4f}")

# Probability of a full sentence
sentence = ["I", "like", "Sam"]

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams

# pad the sentence ONCE
padded = list(pad_both_ends(sentence, n))
# get bigrams
bgs = list(bigrams(padded))

p_sent = 1.0
for w1, w2 in bgs:
    p_sent *= model.score(w2, [w1])

print(f"P({sentence}) = {p_sent:.8f}")

