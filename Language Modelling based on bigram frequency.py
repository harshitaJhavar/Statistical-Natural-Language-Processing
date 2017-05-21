#Assignment 2
#Solution to Question 1

from nltk.corpus import brown #Importing brown corpus from nltk
import matplotlib.pyplot as plt #For plotting
import math #For taking log to calculate expected value

def find_followUp_word_given_word(word):
	'''Making a list of all the words following the given word'''
	next_word = []
	for i in range(len(text)-1):
		if(text[i] == word):
			next_word.append((text[i+1]))
	return next_word

def calculate_relative_frequency(next_word):	
	'''Storing (word following the given word, number of times it follows the given word, relative frequency).  Basically, calculating the conditional probability distribution'''
	relative_frequency = []
	for i in range(len(next_word)-1):
		relative_frequency.append((next_word[i], next_word.count(next_word[i]), (next_word.count(next_word[i]) / float(len(next_word)))))
	#Removing duplicate entries from the list of relative frequency using the set function.
	relative_frequency = list(set(relative_frequency))
	return(relative_frequency)

def find_top_20_tokens(relative_frequency):	
	'''Finding the top 20 words'''
	top_20 = (sorted(relative_frequency, key=lambda relative_frequency: relative_frequency[1], reverse=True))[:20]   # sort by count
	return(top_20)

def plot_frequency_distribution(top_20, word):	
	'''Plotting the required graph'''
	x = []
	y = []
	for i in range(len(top_20)):
		x.append(top_20[i][0])
		y.append(top_20[i][1])
	plt.plot(y)
	plt.xlabel("Top 20 tokens following the word - " +word)
	plt.ylabel("Frequency Distribution ")
	plt.title("Frequency Distribution for top 20-tokens following the word - "+word)
	plt.show()

def calculate_expected_value(relative_frequency):
	'''Computing the expected value for the distributions'''
	expected_value = 0
	for i in range(len(relative_frequency)):
		expected_value = expected_value + ( relative_frequency[i][1] * -1 * math.log(relative_frequency[i][2], 2 ) )#log to the base 2
	return(str(expected_value))


#Tokenizing and lowercasing each token
text = brown.words()
text = [x.lower() for x in text]

#For the word - "in"
next_word_in = find_followUp_word_given_word("in") 
relative_frequency_in = calculate_relative_frequency(next_word_in)
top_20_in = find_top_20_tokens(relative_frequency_in)
plot_frequency_distribution(top_20_in, "in")
print("The expected value for 'in' is " +calculate_expected_value(relative_frequency_in))


#For the word - "the"
next_word_the = find_followUp_word_given_word("the")
relative_frequency_the = calculate_relative_frequency(next_word_the)
top_20_the = find_top_20_tokens(relative_frequency_the)
plot_frequency_distribution(top_20_the, "the")
print("The expected value for 'the' is " +calculate_expected_value(relative_frequency_the))
