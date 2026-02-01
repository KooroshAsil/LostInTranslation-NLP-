import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer


nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(corpus):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in word_tokenize(corpus.lower()) if w.isalnum()]
    filtered_tokens = [w for w in tokens if w not in stop_words]
    final_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    return final_tokens

class BytePairEncoder:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.vocab = None
        self.merges = []

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus_tokens):
        vocab = Counter(corpus_tokens)
        vocab = {" ".join(list(word)) + " </w>": freq for word, freq in vocab.items()}
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)
        self.vocab = vocab

    def encode_word(self, word):
        word = list(word) + ["</w>"]
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            pair_to_merge = None
            for merge in self.merges:
                if merge in pairs:
                    pair_to_merge = merge
                    break
            if pair_to_merge:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == pair_to_merge:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            else:
                break
        return word

    def encode(self, text_tokens):
        return [self.encode_word(word) for word in text_tokens]



if __name__ == "__main__":
    with open("./corpus.txt") as corp:
        corpus = corp.read()

    tokens = preprocess_text(corpus)
    print("Preprocessed corpus tokens:")
    print(tokens)
    print("-"*100)

    bpe = BytePairEncoder(num_merges=100)
    bpe.fit(tokens)  # pass list of words, not joined string

    print("Learned merges:")
    print(bpe.merges)
    print("-"*100)

    with open("./sample_test_text.txt") as test_sample:
        test_corpus = test_sample.read()

    sample_tokens = preprocess_text(test_corpus)
    print("Preprocessed sample text tokens:")
    print(sample_tokens)
    print("-"*100)

    encoded_words = [bpe.encode_word(word) for word in sample_tokens]
    bpe_words = ["".join(word).replace("</w>", "") for word in encoded_words]

    print("BPE Recognized words:")
    print(bpe_words)
