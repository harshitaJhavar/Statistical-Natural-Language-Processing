#!/usr/bin/env python3
#Exercise 2, Task 3
#SNLP, Summer 2017

from nltk import sent_tokenize, wordpunct_tokenize
from collections import defaultdict
import numpy as np # For working with logs
import matplotlib.pyplot as plt

    
def get_counts(text):
    """Returns default dictionaries for unigrams and 
       bigram counts, respectively
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
 

def get_prob_dicts(unigram_dict, bigram_dict, alpha=0.3):
    """Given lists of unigrams and bigrams, and optionally
       alpha value for Lidstone smoothing, returns default
       dictinaries of log-probabilities for unigram and
       bigram language models.
    """
    # Number of types, aka size of vocab
    V = len(unigram_dict) 
    V += 1 # To account for <UNK> marker
    
    # Convert unigram counts to probabilities
    unigram_total = sum(unigram_dict.values()) # Number of tokens
    for word in unigram_dict:
        prob = (unigram_dict[word] + alpha) / (unigram_total + alpha * V)
        unigram_dict[word] = np.log(prob)
    # Add probability mass reserved for unseen tokens
    if alpha > 0:
        unigram_dict["UNK"] = np.log(alpha / (unigram_total + alpha * V))
    else:
        unigram_dict["UNK"] = float("-inf")
        
    # Convert bigram counts to probabilities
    for word1 in bigram_dict:
        word1_total = sum(bigram_dict[word1].values())
        for word2 in bigram_dict[word1]:
            prob = (bigram_dict[word1][word2] + alpha) / (word1_total + alpha * V)
            bigram_dict[word1][word2] = np.log(prob)
        # Add probability mass for unseen word with seen history, eg: P(<UNK>|the)
        prob = alpha / (word1_total + alpha * V)
        bigram_dict[word1]["UNK"] = np.log(prob)
    # Add probability mass for unseen word with unseen history, eg: P(<UNK>|<UNK>)
    # Rough way for handling this situation for now
    if alpha > 0:
        bigram_dict["UNK"] = np.log(1 / V)
    else:
        bigram_dict["UNK"] = float("-inf")
    return unigram_dict, bigram_dict
    

def sanity_check(unigram_probs, bigram_probs):
    """Given smoothed unigram and bigram models,
       checks that probabilities sum to one."""
    probs = list(unigram_probs.values())
    uni_total = sum([np.exp(x) for x in probs])
    print("\tUnigram probability sum:", uni_total)

    word = 'the'
    V = len(vocab)
    bigram_words = len(bigram_probs[word])
    # Number of words not encountered as a bigram with word
    difference = V - bigram_words 
    bi_total = sum([np.exp(bigram_probs[word][x]) for x in bigram_probs[word]])
    bi_total += np.exp(bigram_probs[word]["UNK"]) * difference
    print("\tBigram probability sum (for 'the'): ", bi_total)
    # Round off minor float erros
    if round(uni_total, 5) != 1 or round(bi_total, 5) != 1:
        print("Warning, probabilities do not sum to one!")
    

def get_perplexity(unigram_model, bigram_model, text):
    """Given a unigram and bigram model and a test test,
       calculates the respective perplexities"""
    # Get counts from test text
    unigram_counts, bigram_counts = get_counts(text)
    # Get N values to transform absolute count into relative counts
    unigram_N = sum(unigram_counts.values())
    bigram_N = sum([sum(bigram_counts[word].values()) for word in bigram_counts if word != "UNK"])
    
    # Initialize perplexity value and increment
    unigram_pp = 0
    for word in unigram_counts:
        count = unigram_counts[word]
        if word in unigram_model:
            log_prob = unigram_model[word]
        else:
            log_prob = unigram_model["UNK"]
        unigram_pp += count * log_prob
    unigram_pp = np.exp(-unigram_pp / unigram_N)
    
    # Same process but with bigrams
    bigram_pp = 0
    for word1 in bigram_counts:
        if word1 in bigram_model:
            # For every word that follows a known word
            for word2 in bigram_counts[word1]:
                count = bigram_counts[word1][word2]
                if word2 in bigram_model[word1]:
                    log_prob = bigram_model[word1][word2]
                else:
                    log_prob = bigram_model[word1]["UNK"]
                bigram_pp += count * log_prob
        else: # Unknown word follwed by anything else
            count = sum(bigram_counts[word1].values())
            log_prob = bigram_model["UNK"]
            bigram_pp += count * log_prob
    bigram_pp = np.exp(-bigram_pp / bigram_N)
    return unigram_pp, bigram_pp


if __name__ == "__main__":
    training_filename = "../Materials/English1.txt"
    test_filename = "../Materials/English2.txt"

    with open(training_filename) as f:
        training_text = f.read()
    
    # Check that smoothing is working
    unigram_counts, bigram_counts = get_counts(training_text)
    unigram_probs, bigram_probs = get_prob_dicts(unigram_counts, bigram_counts)
    print("Sanity check for smoothing:")
    sanity_check(unigram_probs, bigram_probs)

    with open(test_filename) as f:
        test_text = f.read()
    print("\nEvaluating against", test_filename.split("/")[-1], "\n")

    fifth = int(len(training_text) / 5)
    stats = []
    for i in range(1,6):
        training_subset = training_text[:(fifth * i)]
        percentage = i * 20
        print("Training on {}% of {}".format(percentage, training_filename.split("/")[-1]))
        # Get count dictionaries
        unigram_counts, bigram_counts = get_counts(training_subset)
        # Create probability dictionaries
        unigram_probs, bigram_probs = get_prob_dicts(unigram_counts, bigram_counts)
        # Calculate perplexity
        unigram_pp, bigram_pp = get_perplexity(unigram_probs, bigram_probs, test_text)
        print("\tUnigram perplexity:", unigram_pp)
        print("\tBigram perplexity: ", bigram_pp, "\n")
        stats += [(percentage, unigram_pp, bigram_pp)]

    x = [x[0] for x in stats]
    y = [x[1:] for x in stats]
    plt.plot(x, y, 'o')
    plt.title("Perplexity Patterns")
    plt.ylabel("Perplexity")
    plt.xlabel("Percentage of training text used")
    plt.legend(["Unigram", "Bigram"])
    plt.show()
