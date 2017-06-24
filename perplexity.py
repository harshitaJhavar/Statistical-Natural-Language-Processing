#!usr/bin/env pytohn3
#Exercise 5, Task 3
#SNLP, Summer 2017

from collections import Counter
from string import punctuation
from collections import defaultdict
from nltk import sent_tokenize, wordpunct_tokenize
import numpy as np
import matplotlib.pyplot as plt


def get_counts(text):
    """Tokenizes text, returning default dictionaries 
       for unigrams and bigram counts, respectively.
       R(h) can later be easily calculated from this structure.
    """
    # Split into sentences
    sents = sent_tokenize(text)
    unigrams = []
    bigrams = []
    for s in sents:
        # Tokenize words, separating out punctuation
        words = wordpunct_tokenize(s)
        # Ignore non-alphabetic entries and make words lowercase
        words = [w.lower() for w in words if w.isalpha()]
        unigrams += words
        bigrams += list(zip(words, words[1:]))
    # Make count dictionaries
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(lambda: defaultdict(int)) # Nested default dictionaries
    for word in unigrams:
        unigram_counts[word] += 1
    for word1, word2 in bigrams:
        bigram_counts[word1][word2] += 1 # eg: word_N-1[word_N] : count
    return unigram_counts, bigram_counts


def unigram_discounting(word, d=0.7):
    """Returns P(w), applying absolute discounting for d > 0.
       For lack of a separate vocab list, just use types from text
    """
    V = len(unigram_counts) # Full vocab 
    n_plus = len(unigram_counts) # Vocab entries encountered in text
    N = sum(unigram_counts.values()) # Number of tokens of vocab words seen in text 
    alpha = d * n_plus / N
    prob = alpha / V    
    if word in unigram_counts:
        # prob += (N(w) - d) / N
        prob += (unigram_counts[word] - d) / N
    return prob


def get_Rh(hist):
    """Count of all bigrams that satisfy N(w,h) > 0, aka
       the number of bigrams that have 'hist' as the history
    """
    if hist in bigram_counts:
        R_h = sum(bigram_counts[hist].values()) 
    else:
        R_h = 0
    return R_h


def bigram_discounting(word, hist, d=0.7):
    """Returns P(w|h), applying absolute discounting for d > 0.
       For unseen history words, falls back to unigram model.
    """
    if hist in bigram_counts:
    # Proper bigram discounting
        prob_w = unigram_discounting(word)
        # Count of the history word
        N_h = unigram_counts[hist] 
        # Count of bigrams starting with the history word
        R_h = get_Rh(hist)
        alpha_h = d * R_h / N_h 
        prob = alpha_h * prob_w
        # If N(w,h) > 0
        if word in bigram_counts[hist]:
            N_wh = bigram_counts[hist][word]
            prob += (N_wh - d) / N_h
    else:
    # Fall back to unigram model
        prob = unigram_discounting(word)
    return prob


def get_perplexity(text, d=0.7):
    """Given a test text, calculates the perplexity.
       Model from training text should already be computed
       outside of the function and stored in global variables
       named unigram_counts and bigram_counts.
       """
    # Get absolute counts of testing text
    unigram_countsTEST, bigram_countsTEST = get_counts(text)
    # Get N values to transform absolute count into relative counts
    bigram_N = sum([sum(bigram_countsTEST[word].values()) for word in bigram_countsTEST])
    # Initialize perplexity value and increment
    bigram_pp = 0
    for word1 in bigram_countsTEST:
        for word2 in bigram_countsTEST[word1]:
            # Probability based on model from training set
            prob = bigram_discounting(word2, word1, d=d) 
            # Note: Will return -inf for unseen words if d=0
            prob = np.log(prob)
            # Times the absolute frequency is seen in testing set
            count = bigram_countsTEST[word1][word2]
            bigram_pp += count * prob
    bigram_pp = np.exp(-bigram_pp / bigram_N)
    return bigram_pp


def fold(full_text, d=0.7, num_folds=5):
    """Applies k-fold cross-validation over a text,
       returning the average perplexity
    """
    all_pp = []
    cutoff = int(len(full_text) / num_folds)
    for i in range(num_folds):
        testing = full_text[i*cutoff:(i+1)*cutoff] # Validation set
        training = full_text[:i*cutoff] + full_text[(i+1)*cutoff:]
        unigram_counts, bigram_counts = get_counts(training)
        pp = get_perplexity(testing, d=d)
        print("Fold", i+1, "perplexity:", pp)
        all_pp += [pp]
    avg_pp = sum(all_pp) / num_folds
    print("Average:", avg_pp)
    return avg_pp


def adjust_d(full_text, num_folds=5):
    """Iteratively applied k-fold cross-validation
       incrementing the d value by 0.1 from 0.0 to 1.0.
    """
    all_pp = []
    for d in range(11):
        d /= 10
        print("\nd =", d)
        pp = fold(full_text, d=d, num_folds=num_folds)
        all_pp += [pp]
    return all_pp


def plot_pp(pp_list):
    """Plots the resulting perplexity values of
       adjusting the discounting parameter
    """
    # Note: Pyplot just skips values that are infinite
    plt.plot(pp_list)
    plt.ylabel("Perplexity")
    plt.xlabel("Discounting Parameter")
    plt.xticks([x for x in range(11)], [x/10 for x in range(11)])
    plt.show()

    
if __name__ == "__main__":
    with open("../materials_ex5/text.txt") as f:
        text = f.read()
    # Global variables reflecting counts from the TRAINING corpus
    unigram_counts, bigram_counts = get_counts(text)
    all_pp = adjust_d(text)
    plot_pp(all_pp)
    min_value = min(all_pp)
    min_d = [i for i in range(len(all_pp)) if all_pp[i] == min_value]
    print("Minimum perplexity at d =", min_d[0]/10)
