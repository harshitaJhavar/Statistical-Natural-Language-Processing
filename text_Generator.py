#!/usr/bin/env python3
#Exercise 1, Quesiton 2
#SNLP, Summer 2017

import string 
import numpy as np
import matplotlib.pyplot as plt


def get_prob_dict(text):
    """Given a text, returns a dictionary containing
       each letter and its relative probability.
    """
    text = text.lower()
    # Make dictionary of relative counts
    count_dict = {}
    for char in text:
        if char.isalpha(): # Ignore spaces, numbers, etc.
            if char in count_dict:
                count_dict[char] += 1 # Increment existing key
            else:
                count_dict[char] = 1 # New key
    total = sum(count_dict.values()) # Number of characters total
    # Generate probability dictionary based on count dictionary
    prob_dict = {char : count_dict[char] / total for char in count_dict}
    return prob_dict


def text_generator(length, space_prob=0.2, unique_char_prob=None):
    """Generates text by randomly selecting characters.
       args:
          length: Number of characters (+whitespace) to generate
          space_prob: Probability of returning a space
          unique_char_prob: Optionally set equal to a text to base probabilities on.
                            When unspecified, defaults to all letters equally likely.
    """
    chars = list(string.ascii_lowercase) # a-z
    if unique_char_prob:
        prob_dict = get_prob_dict(text)
        # Format into a list of probabilities in alphabetic order
        char_probs = []
        for char in chars:
            if char in prob_dict:
                char_probs.append(prob_dict[char])
            else:
                char_probs.append(0)
    output = ""
    prev_space = False # To prevent more than one space in a row
    for i in range(length):
        if np.random.random() <= space_prob and not prev_space:
            output += " "
            prev_space = True
        else:
            if not unique_char_prob: # Each character equally probable
                next_char = np.random.choice(chars) 
            else: # Selects character from probability distribution
                next_char = np.random.choice(chars, p=char_probs)
            output += next_char
            prev_space = False
    return output


def plot_char_prob(text):
    """Plots the frequency of each character for a given text"""
    prob_dict = get_prob_dict(text)
    prob_tuples = [(prob_dict[char], char) for char in prob_dict]
    # Sort by decreasing probability
    prob_tuples = sorted(prob_tuples, reverse=True)
    probs, letters = zip(*prob_tuples)
    plt.plot(probs)
    indecies = list(range(len(letters)))
    plt.xticks(indecies, letters) # Mark letters along x axis
    plt.ylim((0, max(probs) * 1.1))
    plt.ylabel("Probability")
    plt.title("Random Letters Generated based on Miller's Model")
    plt.show()


def zipf_plot(text):
    """Given a text, generates a Zipf plot of word frequencies"""
    text = text.lower()
    text = text.split()
    zipf_dict = {}
    for word in text:
        if word in zipf_dict:
            zipf_dict[word] += 1
        else:
            zipf_dict[word] = 1
    counts = sorted(zipf_dict.values(), reverse=True)
    plt.loglog(counts) # Both axes log scale
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.title("Zipf Plot of Words Generated from Extended Model")
    plt.show()


if __name__ == "__main__":
    print("All letters equally likely:\n")
    random_text_1 = text_generator(100000)
    print("Sample text:\n", random_text_1[:150])
    plot_char_prob(random_text_1)
    with open("../question1/data/english.txt") as f:
        text = f.read()
    english_text = text.replace("\n", " ") # Remove newline markers
    print("Letter probability conditioned on freqency in Othello:\n")
    random_text_2 = text_generator(100000, unique_char_prob=english_text)
    print("Sample text:\n", random_text_2[:150])
    zipf_plot(random_text_2)
