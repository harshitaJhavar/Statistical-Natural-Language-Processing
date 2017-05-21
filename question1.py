#Exercise 3
#Solution to Question 1

import nltk
import string
import math

def pre_process_text(rawtext):
	'''A function to pre-process the text'''
	rawtext = rawtext.lower() #Lowercasing the text
	rawtext = rawtext.translate(str.maketrans('','',string.punctuation)) #Removing the punctuations
	rawtext = nltk.word_tokenize(rawtext) #Tokenize the text
	return(rawtext)

def get_index(prob_distribution,word):
	'''Finding the index of the word in probability distribution list to retrieve its probability from the distribution'''
	for w in range(len(prob_distribution)-1):
		if(prob_distribution[w][0] == word):
			index = w #Getting the index	
	return(index)					
					
def calculate_klDivergence(text1, text2):
	'''A function to calculate the KL Divergence between probability distributions for text1 and text2 to account for 
	the average number of bits that are wasted by encoding events from a distribution of text1 with a code based on a distribution for text2.
	'''
	klDivergence = 0 #Initialising the kldivergence value by zero to store the sum later
	#Assigning the probability distribution to each unigram using its maximum likelihood estimation
	#Probability Distribution of Text 1
	pd_text1 = [] 
	for i in range(len(text1)-1):
		pd_text1.append((text1[i], (text1.count(text1[i]) / float(len(text1)))))
	#Probability Distribution of Text 2
	pd_text2 = [] 
	for j in range(len(text2)-1):
		pd_text2.append((text2[j], (text2.count(text2[j]) / float(len(text2)))))
	#The loop will run for all words in text 1 as we have to find its encoding based on distribution for text2 and then, account for the bits wasted for this encoding.
	for k in range(len(text1)-1):
		#Computing the corresponding probabilities from the distributions for the given word
		#Using Lidstone smoothing as well
		#Here:
		#Probability from the distribution is pd_text1[k][1]
		#The count of the current word in consideration in the given text sample is (text1.count(text1[k])
		#Alpha is 0.1 as given in question
		#len(text1) is the number of tokens after preprocessing the text
		#list(set(text1))).count is the size of vocabulary which will be all the unique list of words without any repetition
		prob_from_text1_distribution = pd_text1[k][1] + ((text1.count(text1[k])+0.1)/(len(text1)+(0.1*(len(set(text1))))))
		#Checking if the current word in consideration is there in the given distribution		
		if(any(e[0] == text1[k] for e in pd_text2)):
			#Finding the index of the current word in text2 probability distribution list to retrieve its probability from text 2 distribution
			ind = get_index(pd_text2,text1[k])
			prob_from_text2_distribution = pd_text2[ind][1] + ((text2.count(text1[k])+0.1)/(len(text2)+(0.1*(len(set(text2)))))) #Final probability with smoothing.
		else:
			prob_from_text2_distribution = 	(text2.count(text1[k])+0.1)/(len(text2)+(0.1*(len(set(text2))))) #Taking care of unknown words to distribution from text2 whose 		prob is zero according to distribution from text2
			
		#Calculating the product for the given word
		klDivergence = klDivergence + (prob_from_text1_distribution * math.log((prob_from_text1_distribution/prob_from_text2_distribution),2))
	return(klDivergence, pd_text1, pd_text2)

def find_followUp_word_given_word(word,text):
	'''Making a list of all the words following the given word i.e making a separate list of bigrams for the word'''
	next_word = []
	for i in range(len(text)-1):
		if(text[i] == word):
			next_word.append((text[i+1]))
	return next_word

def calculate_mutualInformation(text, pd):
	'''Gives the contribution to I(X;Y) by (word1, word2) for the given text and calculates the I(X;Y) for the text'''
	I_text = 0 #Initialising the information value variable to store the sum for each word pair
	Highest_Contribution = 0
	Lowest_Contribution = 0
	for i in range(len(text)-1): 
		for j in range(len(text)-1):
		#The probability P(X|Y) is measured by bigram frequency. Here X is text[j] and Y is text[i]
		#Calculating the bigram frequency for it to assign a probability value to P(X|Y)
			prob_bigram = (text.count(text[i]+' '+text[j])) / float(len(text))
			#Calculating the word pair contribution
			#Computing E[-log P(X)] - E[-log P(X|Y)] for all words in text (Expression taken from part 2 proof of this question)
			# Here P(X) will be pd[j][1].
			if(prob_bigram == 0):
				I_wordpair_contribution = (pd[j][1] * -1 * math.log(pd[j][1],2)) 
			else:
				I_wordpair_contribution = (pd[j][1] * -1 * math.log(pd[j][1],2)) - (prob_bigram * -1 * math.log(prob_bigram,2))	
			I_text = I_text + I_wordpair_contribution #Adding to get the summation of each word contribution to get the final I(X;Y) for the entire text.
			#Checking for highest and lowest
			if(I_wordpair_contribution > Highest_Contribution):
				Highest_Contribution = I_wordpair_contribution
				highestPair = text[j] + ' ' + text[i]
			elif(I_wordpair_contribution < Lowest_Contribution):
				Lowest_contribution = I_wordpair_contribution
				lowestPair = text[j] + ' ' + text[i]
	return(highestPair, lowestPair, Highest_Contribution, Lowest_Contribution, I_text)
			
			
#Importing the required text samples - two in English and the third in German
from nltk.corpus import gutenberg
english1 = nltk.corpus.gutenberg.raw('carroll-alice.txt') #136075 tokens
english2 = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt') #95562 tokens
german1 = nltk.corpus.udhr.raw('German_Deutsch-Latin1') #9836 tokens

#Preprocessing the raw text
english1 = pre_process_text(english1)
english2 = pre_process_text(english2)
german1 = pre_process_text(german1)

#Calculating KL divergence value for different pairs of texts
klDivergence_e1e2, pd_e1, pd_e2 = calculate_klDivergence(english1, english2) #same language
klDivergence_e2e1, pd_e2, pd_e1 = calculate_klDivergence(english2, english1) #same language
klDivergence_e1g1, pd_e1, pd_g1 = calculate_klDivergence(english1, german1) #different language
klDivergence_e2g1, pd_e2, pd_g1 = calculate_klDivergence(english2, german1) #different language
print(klDivergence_e1e2)
print(klDivergence_e2e1)
print(klDivergence_e1g1)
print(klDivergence_e2g1)

#Calculating the Mutual Information for the given probability distributions
high_english1, low_english1, h_e1, l_e1, I_e1 = calculate_mutualInformation(english1, pd_e1)
high_english2, low_english2, h_e2, l_e2, I_e2 = calculate_mutualInformation(english2, pd_e2)
high_german1, low_german1, g_g1, l_g1, I_g1 = calculate_mutualInformation(german1, pd_g1)
print(high_english1) #Word pair contributed highest to English text 1
print(low_english1)	#Word pair contributed lowest to English text 1
print(h_e1)	#Value of highest contribution for English Text 1
print(l_e1)	#Value of lowest contribution for English Text 1
print(I_e1)	#Value of mutual information for English Text 1
print(high_english2)	#Word pair contributed highest to English text 2
print(low_english2)	#Word pair contributed lowest to English text 1
print(h_e2)#Value of highest contribution for English Text 2
print(l_e2)	#Value of lowest contribution for English Text 2
print(I_e2) 	#Value of mutual information for English Text 2
print(high_german1)	#Word pair contributed highest to German text 1
print(low_german1)	#Word pair contributed lowest to German text 1
print(h_g1) #Value of highest contribution for German Text 1
print(l_g1)	#Value of lowest contribution for German Text 1
print(I_g1)		#Value of mutual information for German Text 1


