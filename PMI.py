#!usr/bin/env pytohn3
# Exercise 7, Task 2
# SNLP, Summer 2017

from nltk import wordpunct_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
# One of several options, along with Porer and Lancaster stemmers
from nltk.stem import SnowballStemmer 
import string
from collections import Counter
import numpy as np

wordnet_lemmatizer = WordNetLemmatizer()
snowball_stemmer = SnowballStemmer("english")


def preprocess(text, stopword_list):
    """Preprocesses a given text, removing stopwords 
       (or anything else) passed into the second argument.
    """
    # Make lowercase
    text = text.lower() 
    # Tokenize
    words = wordpunct_tokenize(text)
    output = []
    for word in words:
        # Remove stopwords
        if word not in stopword_list and not word.isdigit():
            # Lemmatize
            word = wordnet_lemmatizer.lemmatize(word)
            # Stem
            word = snowball_stemmer.stem(word)
            output.append(word)
    return output
    

def get_PMI(term, counters):
    """Given a single term and a list of Counter objects 
       for each category, calculates pmi(term, c) for each c,
       returning the results as a list in order of categories.
    """
    results = []
    # Number of tokens over all texts
    total_tokens = sum([sum(c.values()) for c in counters])
    total_count = sum([c[term] for c in counters])
    total_prob = total_count / total_tokens
    # Calculate conditional probability
    for c in counters:
        cond_count = c[term]
        cond_tokens = sum(c.values())
        cond_prob = cond_count / cond_tokens
        pmi = np.log(cond_prob / total_prob)
        results.append(pmi)
    return results
    

if __name__ == "__main__":
    stopword_file = "../Materials_Ex7/stopwords.txt"
    bio_file = "../Materials_Ex7/train/Biology.txt"
    chem_file = "../Materials_Ex7/train/Chemistry.txt"
    phys_file = "../Materials_Ex7/train/Physics.txt"

    with open(stopword_file) as f:
        stopword_list = f.read().split("\n")

    # Each category, in order
    files = [bio_file, chem_file, phys_file]
    counters = []
    vocab = []

    # Preprocess all texts, get count dictionaries for each,
    # and get set of vocabulary
    for file in files:
        with open(file) as f:
            text = f.read()
        text = text.replace("\n", " ")
        print("Preprocessing", file.split("/")[-1])
        text = preprocess(text, stopwords + list(string.punctuation) + ["\ufeff"])
        print(len(text), "tokens\n")
        vocab += text
        counters.append(Counter(text))
    vocab = set(vocab)

    # PMI between each term and each topic
    full_results = []
    # PMI to discriminate well for a single category
    pmi_max = []

    for term in vocab:
        bio, chem, phys = get_PMI(term, counters)
        full_results.append((term, bio, chem, phys))
        # PMI_max value
        highest = max([bio, chem, phys])
        # Which category that corresponds to
        index = np.argmax([bio, chem, phys])
        category = files[index].split("/")[-1].split(".")[0]
        pmi_max.append((term, category, highest))

    # Feature selection
    num_features = 10
    features = sorted(pmi_max, key=lambda n:n[2], reverse=True)[:num_features]

    print("Top", num_features, "features, which category they indicate, and their PMI_max:\n")
    for feat in features:
        print('{0:15}{1:12}{2}'.format(feat[0], feat[1], feat[2]))
