#Solution to question 4
#Exercise 4

#Importing re package to use regular expression as a tokenizer
import re 
#Importing matplotlib for plotting
import matplotlib.pyplot as plt 
#Importing itemgetter for generating the frequency table
from operator import itemgetter 
    
def get_vocabulary(train_File_Path):
	'''
		Returns the vocabulary for the given text and the length of the given text
	'''	
	#Reading the train file
	text_file = open(train_File_Path, 'r')
	#Tokenizing the text using regular expression.
	#The regular expression reads every word whether beginning with a capital or a small letter and the follow up alphabets are marked with question mark to allow words of 		length 1 also like 'I', 'a' etc.
	tokens = re.findall(r'(\b[A-Za-z]+[a-z]?\b)', text_file.read())
	#Lowercasing all the tokens
	tokens = [w.lower() for w in tokens]
	#Finding the length of the texts
	length = len(tokens)
	#Initializing empty dictionary so as to hold the frequency of each different word in the text
	tokens_frequency = {}
	#Calculating the frequency for each word in text
	for word in tokens:
		count = tokens_frequency.get(word,0)
		tokens_frequency[word] = count + 1
    #Sorting the frequencies in reverse order so as to assign a rank to the words from text and generating a lists of words in ascending order of rank.
	vocabulary = []
	for key, value in reversed(sorted(tokens_frequency.items(), key = itemgetter(1))):
		vocabulary.append(key)	
    #Removing duplicate entries from the list of vocabulary using the set function.
	vocabulary = list(set(vocabulary))
	return(vocabulary, length)

def calculate_oov(vocabulary, test_File_Path):
	'''
		Returns the OOV rate value for the given test file with respect to the vocabulary taken as an input
		OOV rate = # unseen words in test corpus / # of tokens in test corpus
	'''
	#length of test corpus
	
	#Using the get_vocabulary function to get a list of all the different unique words in test file and its length
	test_words, length = get_vocabulary(test_File_Path)
	
	#Count for number of unseen words in test corpus
	count = 0
	flag = 0
	for i in range(0,len(test_words)-1):
		for j in range(0, len(vocabulary)-1):
			if(test_words[i]== vocabulary[j]): #Checking if a particular word is in the vocabulary
				flag = 1
				break
		if(flag == 0):
			count = count + 1 #Calculating the number of unseen words
		flag = 0
	return(count / length)				
#Generating vocabulary for all the 5 text files
vocabulary1,l1 = get_vocabulary('./train/train1.txt')
vocabulary2,l2 = get_vocabulary('./train/train2.txt')
vocabulary3,l3 = get_vocabulary('./train/train3.txt')
vocabulary4,l4 = get_vocabulary('./train/train4.txt')
vocabulary5,l5 = get_vocabulary('./train/train5.txt')

#Calculate OOV rate for each vocabulary in reference to the test file
OOV1 = calculate_oov(vocabulary1,'./test/test.txt')
OOV2 = calculate_oov(vocabulary2,'./test/test.txt')
OOV3 = calculate_oov(vocabulary3,'./test/test.txt')
OOV4 = calculate_oov(vocabulary4,'./test/test.txt')
OOV5 = calculate_oov(vocabulary5,'./test/test.txt')
print(OOV1)
print(OOV2)
print(OOV3)
print(OOV4)
print(OOV5)

#Plotting OOV vs Size of Vocabulary
x = [l1,l2,l3,l4,l5]
y = [OOV1,OOV2,OOV3,OOV4,OOV5]
plt.xlabel('Size of Vocabulary')
plt.ylabel('OOV rate')
plt.title("Plot of OOV rate vs Size of Vocabulary")
plt.plot(x,y)
plt.show()

