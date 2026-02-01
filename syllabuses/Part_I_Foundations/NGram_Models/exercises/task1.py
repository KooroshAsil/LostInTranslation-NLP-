
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize import word_tokenize
from nltk import FreqDist
    
    
"""
TASK 1 
   get all bigrams and unigrams
   
"""
    
    
def filter_invalid_ngrams(nested_list):
    """
    Filters a nested list of n-gram tuples to remove invalid combinations.

    An n-gram is considered invalid if it contains consecutive '<s>' or '</s>' tokens.
    For example: ('<s>', '<s>'), ('<s>', '<s>', 'i'), ('sam', '</s>', '</s>')

    Args:
        nested_list: A list of lists, where each inner list contains tuples of strings.

    Returns:
        A new nested list with the invalid n-grams removed.
    """
    filtered_nested_list = []
    for inner_list in nested_list:
        filtered_inner_list = []
        for ngram_tuple in inner_list:
            is_valid = True
            # Check for consecutive <s> or </s> tokens
            for i in range(len(ngram_tuple) - 1):
                if (ngram_tuple[i] == '<s>' and ngram_tuple[i+1] == '<s>') or \
                   (ngram_tuple[i] == '</s>' and ngram_tuple[i+1] == '</s>'):
                    is_valid = False
                    break
            
            if is_valid:
                filtered_inner_list.append(ngram_tuple)
        
        filtered_nested_list.append(filtered_inner_list)
        
    return filtered_nested_list

def TextFile_Corpus(path = "./corpus.txt"):
    
    """
    Reads a text file and prepares a corpus by converting all text to lowercase 
    and stripping any leading or trailing whitespace from each line.

    This function takes the path to a text file as input, reads its contents, 
    and processes each line to create a list of sentences. Each sentence is 
    converted to lowercase to ensure uniformity, making it easier to analyze 
    the text later on.

    Parameters:
    ----------
    path : str, optional
        The file path to the text file containing the corpus. The default value 
        is "./corpus.txt", which means it will look for a file named 'corpus.txt' 
        in the current directory. You can specify a different path if your file 
        is located elsewhere.

    Returns:
    -------
    corpus : list of str
        A list of strings, where each string represents a line from the text file. 
        Each line is processed to be in lowercase and stripped of any extra spaces. 
        This list serves as the basis for further text analysis or natural language 
        processing tasks.
        
    """
    with open(path, 'r', encoding='utf-8') as f:
        corpus = [s.strip().lower() for s in f.readlines()]
    return corpus


corpus = TextFile_Corpus(path="./corpus.txt")

def get_Ngrams(train_data, n):
    Ngrams = [[gram for gram in train_data[i] if len(gram) == n] for i in range(len(train_data)) ]
    return Ngrams

def prepare_corupus (corpus, n = 1):
    
    """
    Prepares a text corpus for n-gram modeling by tokenizing the sentences, 
    generating n-grams, and organizing them into various structures.

    This function takes a list of sentences (the corpus) and processes it to 
    create n-grams of specified lengths. It tokenizes the sentences, converts 
    them to lowercase, and generates padded n-grams. The output includes unique 
    n-grams, flattened n-grams, and the original n-grams in their nested form, 
    along with the training data and vocabulary.

    Parameters:
    ----------
    corpus : list of str
        A list of sentences (strings) that make up the text corpus. Each sentence 
        should be a coherent string of words. For example:
        [
            "I am Sam",
            "Sam I am",
            "Sam I like",
            "I do like Sam",
            "do I like Sam"
        ]

    n : int, optional
        The maximum length of n-grams to generate. The default is 1, which means 
        the function will generate unigrams. If set to 2, it will generate 
        bigrams, and so on. This allows for flexibility in the type of n-grams 
        you want to analyze.

    Returns:
    -------
    unique_n_grams : dict
        A dictionary where each key corresponds to the n-gram length (e.g., 
        '1-grams', '2-grams') and the value is a list of unique n-grams of that 
        length found in the corpus. This helps in understanding the distinct 
        phrases present in the text.

    n_grams_flatten : dict
        A dictionary similar to `unique_n_grams`, but the values are flattened 
        lists of n-grams. This means that all n-grams of a particular length 
        are combined into a single list, making it easier to analyze the 
        frequency of occurrences.

    n_grams : dict
        A dictionary containing the original nested structure of n-grams. Each 
        key corresponds to the n-gram length, and the value is a list of lists, 
        where each inner list contains the n-grams for a specific sentence in 
        the corpus. This structure retains the context of where each n-gram 
        originated.

    train_data : list of list of tuples
        A list of tuples representing the padded n-grams generated from the 
        tokenized sentences. This is the data that can be used to train an 
        n-gram language model.

    vocab : list
        A list of unique words (tokens) found in the corpus, which serves as the 
        vocabulary for the n-gram model. This is useful for understanding the 
        diversity of the language used in the corpus.

    all_words : list
        A list of all words (tokens) from the corpus, including duplicates. This 
        provides insight into the overall word usage in the text.

    """    
    tokenized = [[w.lower() for w in word_tokenize(sent)] for sent in corpus]

    train_data, vocab = padded_everygram_pipeline(n, tokenized)
    all_words = list(vocab)
    train_data = [list(g) for g in list(train_data)]
    train_data = filter_invalid_ngrams(train_data)
    
    vocab = list(set(all_words))
    
    n_grams = {}
    n_grams_flatten = {}
    unique_n_grams = {}
    
    for j in range(n):
        n_grams[f'{j+1}-grams'] = get_Ngrams(train_data, j+1)
        
    for k in range(n):
        n_grams_flatten[f'{k+1}-grams'] = [item for sublist in n_grams[f'{k+1}-grams'] for item in sublist]

    for t in range(n):
        unique_n_grams[f'{t+1}-grams'] = list(set(n_grams_flatten[f'{t+1}-grams']))
            
    return unique_n_grams, n_grams_flatten, n_grams, train_data, vocab, all_words


uninq_n_grams, n_grams_flatten, n_grams,train_data, vocab, all_words = prepare_corupus(corpus, n=2)


