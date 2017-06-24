#!/usr/bin/env python3
#Exercise 4, Task 4
#SNLP, Summer 2017

import nltk
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def normalize(text, save=True):
    text = text.lower()
    # Remove punctuation and newline characters
    from string import punctuation
    for punc in punctuation:
        text = text.replace(punc, "")
    text = text.replace("\n", " ")
    # Normalize forms of you
    other_forms = ["you'll", "your", "you're", "you've", "you'd", "yours", "yourself", "yourselves"]
    for variant in other_forms:
        text = text.replace(variant, "you")
    if save:  
        with open("normalized.txt", "w") as f:
            f.write(text)
        print("Saved as normalized.txt")
    return text


def find_correlation(text, target_word="you", max_dist=50):
    """Given an unsplit text, finds the autocorrelation values
       for a target word from distance 1 to max_dist, returning
       a list of tuples (dist, correlation).
    """
    text = text.split()
    count = Counter(text)[target_word] # N(w)
    prob = count / len(text) # P(w)
    results = []
    for dist in range(1, max_dist + 1):
        # Get count: N_dist(w w)
        dist_seen = 0
        for i in range(len(text) - dist):
            if text[i] == target_word:
                if text[i + dist] == target_word:
                    dist_seen += 1
        # Get probability: P_dist(w w)
        dist_prob = dist_seen / (len(text) - dist)
        # c_dist(w) = P_dist(w w)/P(w)^2
        correlation = dist_prob / prob ** 2
        results.append((dist, correlation))
    return results


def plot_correlation(data, log_x_axis=False):
    """Given a list of tuples (distance, correlation) 
       tuples, draws plot"""
    x,y = zip(*data)
    if not log_x_axis:
        # Straight line at y = 1 for reference
        plt.plot(np.ones(len(y) + 1)) 
        plt.plot(x,y)
    else:
        plt.semilogx(np.ones(len(y) + 1)) 
        plt.semilogx(x,y)
    plt.xlabel("Distance (1 = Bigram)")
    plt.ylabel("Auto-correlation")
    plt.title("Correlation for Forms of 'you' in Text")
    plt.show()
    
    
if __name__ == "__main__":
    # Get text, lowercase, remove punctuation, normalize forms of "you"
    text = nltk.corpus.gutenberg.raw('austen-emma.txt') 
    text = normalize(text)
    # Calculate and graph correlation values
    results = find_correlation(text, max_dist=50)
    plot_correlation(results)
    print("Also looking up to distance 1000:")
    results = find_correlation(text, max_dist=1000)
    plot_correlation(results, log_x_axis=True)
