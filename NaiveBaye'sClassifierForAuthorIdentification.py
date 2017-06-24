#Exercise 07
#Question 1
#Naive Bayes Classifier for author identification

from collections import defaultdict
from string import punctuation
from decimal import *
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
    
#Reading and preprocessing(removing punctuations and lowercasing capital case) the data files
t1_a1 = strip_punctuation(open("./data/author1/pg121.txt","r").read()).lower()
t2_a1 = strip_punctuation(open("./data/author1/pg158.txt","r").read()).lower()
t3_a1 = strip_punctuation(open("./data/author1/pg1342.txt","r").read()).lower()
t1_a2 = strip_punctuation(open("./data/author2/pg76.txt","r").read()).lower()
t2_a2 = strip_punctuation(open("./data/author2/pg119.txt","r").read()).lower()
t3_a2 = strip_punctuation(open("./data/author2/pg3176.txt","r").read()).lower()

#Total author 1 text
t_a1 = (t1_a1 + t2_a1 + t3_a1)
length_t_a1 = len(t_a1)

#Total author 2 text
t_a2 = (t1_a2 + t2_a2 + t3_a2)
length_t_a2 = len(t_a2)

#Calculating the frequency of each word with respect to different authors

#Word frequency in three documents of Author1
wordcount_a1 = defaultdict(int)
for word in t_a1.split():
	wordcount_a1[word] += 1
	
#Word frequency in three documents of Author2
wordcount_a2 = defaultdict(int)
for word in t_a2.split():
	wordcount_a2[word] += 1

#Word frequency in all six documents or entire training set
wordcount_all = defaultdict(int)
for word in (t_a1+t_a2).split():
	wordcount_all[word] += 1
	
#Reading the test corpus and preprocessing it
t1 = strip_punctuation(open("./data/test_author/test1.txt","r").read()).lower()
t2 = strip_punctuation(open("./data/test_author/test2.txt","r").read()).lower()
t3 = strip_punctuation(open("./data/test_author/test3.txt","r").read()).lower()

#Calculating the posterior probability for each sample test text and concluding the author
posterior_probability_a1 = 1
posterior_probability_a2 = 1
alpha = 0.3
#For test set 1
for word in t1.split():
	posterior_probability_a1 *= Decimal( ((wordcount_a1[word] + alpha)/ (length_t_a1 + alpha * len(wordcount_all) ) ) * ((length_t_a1 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) )) )) #Lindstone smoothing in all
	posterior_probability_a2 *= Decimal( ((wordcount_a2[word] + alpha)/ (length_t_a2 + alpha * len(wordcount_all) ) )* ((length_t_a2 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) ) )))
print("For test text 1")
print("posterior_probability for a1 is" , posterior_probability_a1)
print("posterior_probability for a2 is" , posterior_probability_a2)
if 	posterior_probability_a1 > posterior_probability_a2:
	print("author 1 is the author ")
else:
	print("author 2 is the author")

#For test set 2
posterior_probability_a1 = 1
posterior_probability_a2 = 1
for word in t2.split():
	posterior_probability_a1 *= Decimal(((wordcount_a1[word] + alpha)/ (length_t_a1 + alpha * len(wordcount_all) ) ) * ((length_t_a1 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) )) )) #Lindstone smoothing in all

	posterior_probability_a2 *= Decimal(((wordcount_a2[word] + alpha)/ (length_t_a2 + alpha * len(wordcount_all) ) )* ((length_t_a2 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) ) )))
print("For test text 2")
print("posterior_probability for a1 is" , posterior_probability_a1)
print("posterior_probability for a2 is" , posterior_probability_a2)
if 	posterior_probability_a1 > posterior_probability_a2:
	print("author 1 is the author ")
else:
	print("author 2 is the author")
#For test set 3
posterior_probability_a1 = 1
posterior_probability_a2 = 1
for word in t3.split():
	posterior_probability_a1 *= Decimal(((wordcount_a1[word] + alpha)/ (length_t_a1 + alpha * len(wordcount_all) ) ) * ((length_t_a1 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) ))) ) #Lindstone smoothing in all
	posterior_probability_a2 *= Decimal(((wordcount_a2[word] + alpha)/ (length_t_a2 + alpha * len(wordcount_all) ) )* ((length_t_a2 /(length_t_a1 + length_t_a2) )/ ((wordcount_all[word] + alpha )/(length_t_a1 + length_t_a2 + alpha * len(wordcount_all) ) )))
	
	
print("For test text 3")
print("posterior_probability for a1 is" , posterior_probability_a1)
print("posterior_probability for a2 is" , posterior_probability_a2)
if 	posterior_probability_a1 > posterior_probability_a2:
	print("author 1 is the author ")
else:
	print("author 2 is the author")	
