#Question 4
#SNLP Exercise 07

import nltk #To preprocess the text
from nltk.stem import SnowballStemmer 
from nltk.stem.wordnet  import WordNetLemmatizer
from string import punctuation
from collections import defaultdict
import math
from decimal import *
from itertools import islice
from sklearn.cluster import KMeans
import operator
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def contains_word(s, w):
    return (' ' + w + ' ') in (' ' + s + ' ')
        
def preprocess_text(filename):
	#Preprocesses a file
		#Snowball Stemmer
	snowball_stemmer = SnowballStemmer("english")
	stemmered_wordlist = snowball_stemmer.stem(filename)
	#WordNetLemmatizer
	lmtzr = WordNetLemmatizer()
	cleaned_wordlist = lmtzr.lemmatize(stemmered_wordlist)
	tokens = nltk.word_tokenize(cleaned_wordlist) #Tokenizing
	processed_word_list = [] #Removing stop words
	for word in tokens:
		if word not in stop_words:
			processed_word_list.append(word)
	
	return(' '.join(processed_word_list))

#Reading the train files
biology_text = strip_punctuation(open("./data/train/Biology.txt","r").read().lower())
chemistry_text = strip_punctuation(open("./data/train/Chemistry.txt","r").read().lower())
physics_text = strip_punctuation(open("./data/train/Physics.txt","r").read().lower())
#Reading the stop words file
stop_words = open("./data/stopwords.txt","r").read().lower()
#Reading the test file
test_text = open("./data/test_2.txt","r").read().lower()
test_text = preprocess_text(test_text)
	
#Preprocessing the train files
b_text = preprocess_text(biology_text)
c_text = preprocess_text(chemistry_text)
p_text = preprocess_text(physics_text)

#Computing the tfidf taking all three documents text into consideration

#tf value in train text from all three documents into consideration

#tf values in train text for Biology
tf_b = defaultdict(int)
for word in (b_text).split():
	tf_b[word] += 1

#tf values in train text for Chemistry
tf_c = defaultdict(int)
for word in (c_text).split():
	tf_c[word] += 1

#tf values in train text for Physics
tf_p = defaultdict(int)
for word in (p_text).split():
	tf_p[word] += 1

#idf value for Biology class
idf_b = defaultdict(int)
count = 0
for word in (b_text).split():
	if contains_word(b_text, word):
		count += 1
	if contains_word(c_text, word):
		count += 1
	if contains_word(p_text, word):
		count += 1
	idf_b[word] = math.log(3/count) #Total number of documents is three.

#idf value for physics class
idf_p = defaultdict(int)
count = 0
for word in (p_text).split():
	if contains_word(b_text, word):
		count += 1
	if contains_word(c_text, word):
		count += 1
	if contains_word(p_text, word):
		count += 1
	idf_p[word] = math.log(3/count) #Total number of documents is three.

#idf value for Chemistry class 
idf_c = defaultdict(int)
count = 0
for word in (c_text).split():
	if contains_word(b_text, word):
		count += 1
	if contains_word(c_text, word):
		count += 1
	if contains_word(p_text, word):
		count += 1
	idf_c[word] = math.log(3/count) #Total number of documents is three.

#Assigning the tfidf value
#Biology
tfidf_b = defaultdict(int)
for word in (b_text).split():
	tfidf_b[word] = tf_b[word] * idf_b[word] 
#Class- Chemistry
tfidf_c = defaultdict(int)
for word in (c_text).split():
	tfidf_c[word] = tf_c[word] * idf_c[word] 
#Class- Physics			
tfidf_p = defaultdict(int)
for word in (p_text).split():
	tfidf_p[word] = tf_p[word] * idf_p[word] 
#Sorting to get top 500 tfidf values
featurelist_tfidf_b = defaultdict(int)
featurelist_tfidf_c = defaultdict(int)
featurelist_tfidf_p = defaultdict(int)
featurelist_tfidf_b = sorted(tfidf_b.items(), key=operator.itemgetter(1),reverse=True)[:500]
featurelist_tfidf_c = sorted(tfidf_b.items(), key=operator.itemgetter(1),reverse=True)[:500]
featurelist_tfidf_p = sorted(tfidf_b.items(), key=operator.itemgetter(1),reverse=True)[:500]

#Applying the Multimodial Naive Bayes Classification

#Calculating the posterior probability for each sample test text and concluding the author
posterior_probability_biology = 1
posterior_probability_physics = 1
posterior_probability_chemistry = 1
alpha = 0.3

#Calculating the vocabulary size
wordcount_all = defaultdict(int)
for word in (b_text + c_text + p_text).split():
	wordcount_all[word] += 1
vocabularySize = len(wordcount_all)	
#For test set
for word in test_text.split():
	 for pair in featurelist_tfidf_b: 
	 	if(pair[0] == word):
	 		b_tfidf = pair[1]
	 	else:
	 		b_tfidf = 0	
	 for pair in featurelist_tfidf_c: 
	 	if(pair[0] == word):
	 		c_tfidf = pair[1]
	 	else:
	 		c_tfidf = 0
	 for pair in featurelist_tfidf_p: 
	 	if(pair[0] == word):
	 		p_tfidf = pair[1]
	 	else:
	 		p_tfidf = 0
	#Applying the Naive Bayes Classifier here 		
	 posterior_probability_biology *= Decimal( ((b_tfidf + alpha)/ (len(b_text) + alpha * vocabularySize ) ) * (((len(b_text) /len(b_text + c_text + p_text) )/ ((wordcount_all[word] + alpha )/(len(b_text + c_text + p_text) + alpha * vocabularySize )) )) )#Lindstone smoothing in all
	 posterior_probability_physics *= Decimal( ((p_tfidf + alpha)/ (len(p_text) + alpha * vocabularySize ) )* (((len(p_text) /len(b_text + c_text + p_text) )/ ((wordcount_all[word] + alpha )/(len(b_text + c_text + p_text) + alpha * vocabularySize ) ))))
	 posterior_probability_chemistry *= Decimal( ((c_tfidf + alpha)/ (len(c_text) + alpha * vocabularySize ) )* (((len(c_text) /len(b_text + c_text + p_text) )/ ((wordcount_all[word] + alpha )/(len(b_text + c_text + p_text) + alpha * vocabularySize ) ))))
print("For test text with tfidf in Naive bayes Classifier")
print("posterior_probability for biology is" , posterior_probability_biology)
print("posterior_probability for physics is" , posterior_probability_physics)
print("posterior_probability for chemistry is" , posterior_probability_chemistry)
if 	posterior_probability_biology == max(posterior_probability_biology, posterior_probability_chemistry, posterior_probability_physics):
	print("biology is the subject ")
elif posterior_probability_chemistry == max(posterior_probability_biology, posterior_probability_chemistry, posterior_probability_physics):
	print("chemistry is the subject")
else:	
	print("physics is the subject")

#Applying the kmeans algorithm

#Using the  frequency of the words as values of each feature in the feature vector of each document
count_b = 0
count_p =0
count_c = 0
for word in test_text:
	if contains_word(b_text, word):
		count_b += 1
	if contains_word(c_text, word):
		count_c += 1
	if contains_word(p_text, word):
		count_p += 1
print("k-means with  frequency of the words as values of each feature in the feature vector of each document gives ")
if 	count_b == max(count_b, count_c, count_p):
	print("biology is the subject ")
elif count_c == max(count_b, count_c, count_p):
	print("chemistry is the subject")
else:	
	print("physics is the subject")				
# Using tfidf value as values of each feature in the feature vector of each document
count_b = 0
count_p =0
count_c = 0
for word in test_text:
	if contains_word(tf_b, word):
		count_b += tfidf_b[word]
	if contains_word(tf_c, word):
		count_c += tfidf_c[word]
	if contains_word(tf_p, word):
		count_p += tfidf_p[word]
print("k-means with  tfidf value as values of each feature in the feature vector of each document gives")
if 	count_b == max(count_b, count_c, count_p):
	print("biology is the subject ")
elif count_c == max(count_b, count_c, count_p):
	print("chemistry is the subject")
else:	
	print("physics is the subject")	
