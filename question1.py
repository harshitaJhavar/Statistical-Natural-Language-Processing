#Solution of question1 of exercise 1

#Importing re package to use regular expression as a tokenizer
import re 
#Importing matplotlib for plotting
import matplotlib.pyplot as plt 
#Importing itemgetter for generating the frequency table
from operator import itemgetter 

#Reading all three text files
english_file = open('./data/english.txt', 'r')
french_file = open('./data/french.txt', 'r')
german_file = open('./data/german.txt', 'r')

#Tokenizing the three texts using regular expression.
#The regular expression reads every word whether beginning with a capital or a small letter and the follow up alphabets are marked with question mark to allow words of length 1 also like 'I', 'a' etc.
english_tokens = re.findall(r'(\b[A-Za-z]+[a-z]?\b)', english_file.read())
french_tokens = re.findall(r'(\b[A-Za-z]+[a-z]?\b)', french_file.read())
german_tokens = re.findall(r'(\b[A-Za-z]+[a-z]?\b)', german_file.read())

#Initializing three empty dictionaries so as to hold the frequency of each different word in the texts
english_frequency = {}
french_frequency = {}
german_frequency = {}

#Calculating the frequency for each word in English
for word in english_tokens:
    english_count = english_frequency.get(word,0)
    english_frequency[word] = english_count + 1
    
#Calculating the frequency for each word in French
for word in french_tokens:
    french_count = french_frequency.get(word,0)
    french_frequency[word] = french_count + 1
    
#Calculating the frequency for each word in German   
for word in german_tokens:
    german_count = german_frequency.get(word,0)
    german_frequency[word] = german_count + 1

#Sorting the frequencies in reverse order so as to assign a rank to the words in all three different texts and generating a lists in form of pair (word, frequency) in ascending order of rank.
english_ranks = []
german_ranks = []
french_ranks = []
english_table = []
german_table = []
french_table = []

#For English text
for key, value in reversed(sorted(english_frequency.items(), key = itemgetter(1))):
    english_table.append([key, value])
    english_ranks.append(value)
    
#For French text
for key, value in reversed(sorted(french_frequency.items(), key = itemgetter(1))):
    french_table.append([key, value])
    french_ranks.append(value)
    
#For German text
for key, value in reversed(sorted(german_frequency.items(), key = itemgetter(1))):
    german_table.append([key, value])
    german_ranks.append(value)

#Printing the generated word frequency table as mentioned in the question.
#For English
print("Word   ::  Frequency")
for item in english_table:
	print(item[0],"  ::  ",item[1]) 
#For French
print("Word   ::  Frequency")
for item in french_table:
	print(item[0],"  ::  ",item[1]) 
#For German	    
print("Word   ::  Frequency")
for item in german_table:
	print(item[0],"  ::  ",item[1]) 
	  
#Plotting the frequencies on a logarithmic scale
plt.xscale('log')
plt.yscale('log')
e, = plt.plot(english_ranks, color = 'red', label='English Text Word Frequency')
f, = plt.plot(french_ranks, color = 'blue', label='French Text Word Frequency')
g, = plt.plot(german_ranks, color = 'green', label='German Text Word Frequency')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf's Law on Othello by Shakespeare")
plt.legend(handles=[e,f,g])
plt.show()        

