from nltk.corpus import senseval
from nltk import FreqDist

import numpy as np

hard_f, interest_f, line_f, serve_f = senseval.fileids()

def senses(word):
    """
    Returns the list of possible 
    senses for the ambiguous word
    """
    return list(set(i.senses[0] for i in senseval.instances(word)))

def sense_instances(instances, sense):
    """
    This returns the list of instances in instances that have the sense sense
    """
    return [instance for instance in instances if instance.senses[0]==sense]
    
STOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 'also', 'an', 'any',
             'are', 'as', 'at', 'and', 'be', 'being', 'because', 'been', 'but', 'by',
             'can', "'d", 'did', 'do', "don'", 'don', 'for', 'from', 'had','has', 'have', 'he',
             'her','him', 'his', 'how', 'if', 'is', 'in', 'it', 'its', "'ll", "'m", 'me',
             'more', 'my', 'n', 'no', 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s",
             'said', 'say', 'says', 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the',
             'them', 'they', 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you',
             'your']

STOPWORDS_SET=set(STOPWORDS)                                 

def find_top10_mostfrequent_contextwords(instances, stopwords=STOPWORDS_SET, n=10):
    """
    Given a list of senseval instances, return a list of the 10 most frequent words that
    appears in its context
    """
    fd = FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        words = (c[0] for c in i.context if not c[0] == target)
        for word in set(words) - set(stopwords):
            fd[word] += 1           
    return (fd.most_common()[:n+1], fd.most_common()[:])

def pmi(indicators, corpus_sense,  total_top10_indicators_sense_for_different_meanings, total_corpus_from_all_different_senses_for_ambiguous_word):
	'''Returns the pmi for all indicators corresponding to particular sense'''	
	pmi = []
	tokens = []
	for word in total_top10_indicators_sense_for_different_meanings:
		#Calculating pmi for all possible indicator words	
		if([i for i, v in enumerate(indicators) if v[0][0] == word[0][0]]):
			joint_probability = (indicators[[p[0][0] for p in indicators].index(word[0][0])][1] / len(total_corpus_from_all_different_senses_for_ambiguous_word))
			pmi.append(joint_probability * np.log ((joint_probability/len(corpus_sense))/((indicators[[p[0][0] for p in indicators].index(word[0][0])][1])/len(total_corpus_from_all_different_senses_for_ambiguous_word))))
		else:
			pmi.append(0)		
	return(pmi)

#Finding the different word senses for each target label which forms the different classes of translation for each ambiguous word
hard_senses = senses('hard.pos')
line_senses = senses('line.pos')
serve_senses = senses('serve.pos')
interest_senses = senses('interest.pos')  
print("The different senses for each ambiguous word given in the exercise are:")
print(hard_senses)
print(line_senses)
print(serve_senses)
print(interest_senses)

#Finding top 10 indicator words for all ambiguous words corresponding to all their different translations excluding the stop words.
#For Hard1
indicator_words_hard1, corpus_hard1 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('hard.pos'), 'HARD1')), n=10)
#print(indicator_words_hard1)
#For Hard2
indicator_words_hard2, corpus_hard2 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('hard.pos'), 'HARD2')), n=10)
#print(indicator_words_hard2)
#For Hard3
indicator_words_hard3, corpus_hard3 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('hard.pos'), 'HARD3')), n=10)
#print(indicator_words_hard3)
#For 'formation'
indicator_words_formation, corpus_formation = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'formation')), n=10)
#print(indicator_words_formation)
#For 'division'
indicator_words_division, corpus_division = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'division')), n=10)
#print(indicator_words_division)
#For 'text'
indicator_words_text, corpus_text = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'text')), n=10)
#print(indicator_words_text)
#For 'phone'
indicator_words_phone, corpus_phone = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'phone')), n=10)
#print(indicator_words_phone)
#For 'product'
indicator_words_product, corpus_product = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'product')), n=10)
#print(indicator_words_product)
#For 'cord'
indicator_words_cord, corpus_cord = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('line.pos'), 'cord')), n=10)
#print(indicator_words_cord)
#For 'SERVE6'
indicator_words_SERVE6, corpus_SERVE6 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('serve.pos'), 'SERVE6')), n=10)
#print(indicator_words_SERVE6)
#For 'SERVE10'
indicator_words_SERVE10, corpus_SERVE10 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('serve.pos'), 'SERVE10')), n=10)
#print(indicator_words_SERVE10)
#For 'SERVE12'
indicator_words_SERVE12, corpus_SERVE12 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('serve.pos'), 'SERVE12')), n=10)
#print(indicator_words_SERVE12)
#For 'SERVE2'
indicator_words_SERVE2, corpus_SERVE2 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('serve.pos'), 'SERVE2')), n=10)
#print(indicator_words_SERVE2)
#For 'interest_6'
indicator_words_interest_6, corpus_interest_6 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_6')), n=10)
#print(indicator_words_interest_6)
#For 'interest_1'
indicator_words_interest_1, corpus_interest_1 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_1')), n=10)
#print(indicator_words_interest_1)
#For 'interest_4'
indicator_words_interest_4, corpus_interest_4 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_4')), n=10)
#print(indicator_words_interest_4)
#For 'interest_3'
indicator_words_interest_3, corpus_interest_3 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_3')), n=10)
#print(indicator_words_interest_3)
#For 'interest_2'
indicator_words_interest_2, corpus_interest_2 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_2')), n=10)
#print(indicator_words_interest_2)
#For 'interest_5'
indicator_words_interest_5, corpus_interest_5 = find_top10_mostfrequent_contextwords((sense_instances(senseval.instances('interest.pos'), 'interest_5')), n=10)
#print(indicator_words_interest_5)
#Total set of top 10 indicators corresponding to each sense
total_top10_indicators_sense_hard = indicator_words_hard1 + indicator_words_hard2 + indicator_words_hard3
total_top10_indicators_sense_line = indicator_words_formation + indicator_words_division + indicator_words_text + indicator_words_phone + indicator_words_product + indicator_words_cord
total_top10_indicators_sense_serve = indicator_words_SERVE6 + indicator_words_SERVE10 + indicator_words_SERVE12 + indicator_words_SERVE2
total_top10_indicators_sense_interest = indicator_words_interest_6 + indicator_words_interest_1 + indicator_words_interest_4 + indicator_words_interest_3 + indicator_words_interest_2+indicator_words_interest_5
#Combining the total corpus for each of different ambiguous words
total_corpus_from_all_different_senses_HARD = corpus_hard1 + corpus_hard2 + corpus_hard3
total_corpus_from_all_different_senses_LINE = corpus_formation + corpus_division + corpus_text + corpus_phone +  corpus_product + corpus_cord
total_corpus_from_all_different_senses_SERVE = corpus_SERVE6 + corpus_SERVE10 + corpus_SERVE12 + corpus_SERVE2
total_corpus_from_all_different_senses_INTEREST = corpus_interest_6 + corpus_interest_1 + corpus_interest_4 + corpus_interest_3 + corpus_interest_2 + corpus_interest_5

#Calculating the partition for each sense
partition_hard1 = []
partition_hard2 = []
partition_hard3 = []
partition_formation = []
partition_division = []
partition_text = []
partition_phone = []
partition_product = []
partition_cord = []
partition_SERVE6 = []
partition_SERVE10 = []
partition_SERVE12 = []
partition_SERVE2 = []
partition_interest_6 = []
partition_interest_1 = []
partition_interest_4 = []
partition_interest_3 = []
partition_interest_2 = []
partition_interest_5 = []

#For ambiguous word "Hard"
hard1 = pmi(indicator_words_hard1, corpus_hard1, total_top10_indicators_sense_hard, total_corpus_from_all_different_senses_HARD)
hard2 = pmi(indicator_words_hard2, corpus_hard2, total_top10_indicators_sense_hard, total_corpus_from_all_different_senses_HARD)
hard3 = pmi(indicator_words_hard3, corpus_hard3, total_top10_indicators_sense_hard, total_corpus_from_all_different_senses_HARD)
#Applying Flip-Flop
for i in range(0,len(total_top10_indicators_sense_hard)-1):
	if(hard1[i] == max(hard1[i], hard2[i], hard3[i])):
		partition_hard1.append(total_top10_indicators_sense_hard[i][0])
	elif(hard2[i] == max(hard1[i], hard2[i], hard3[i])):
		partition_hard2.append(total_top10_indicators_sense_hard[i][0])
	else:
		partition_hard3.append(total_top10_indicators_sense_hard[i][0])
print(partition_hard1, " is partition for hard 1")
print(partition_hard2, " is partition for hard 2")
print(partition_hard3, " is partition for hard 3")

#For ambiguous word "line"
formation = pmi(indicator_words_formation, corpus_formation, total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
division = pmi(indicator_words_division, corpus_division,  total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
text = pmi(indicator_words_text, corpus_text,  total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
phone = pmi(indicator_words_phone, corpus_phone,  total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
product = pmi(indicator_words_product, corpus_product,  total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
cord = pmi(indicator_words_cord, corpus_cord,  total_top10_indicators_sense_line, total_corpus_from_all_different_senses_LINE)
#Applying Flip-Flop
for i in range(0,len(total_top10_indicators_sense_line)-1):
	if(formation[i] == max(formation[i], division[i], text[i], phone[i], product[i], cord[i])):
		partition_formation.append(total_top10_indicators_sense_line[i][0])
	elif(division[i] == max(formation[i], division[i], text[i], phone[i], product[i], cord[i])):
		partition_division.append(total_top10_indicators_sense_line[i][0])
	elif(text[i] == max(formation[i], division[i], text[i], phone[i], product[i], cord[i])):	
		partition_text.append(total_top10_indicators_sense_line[i][0])	
	elif(phone[i] == max(formation[i], division[i], text[i], phone[i], product[i], cord[i])):		
		partition_phone.append(total_top10_indicators_sense_line[i][0])
	elif(product[i] == max(formation[i], division[i], text[i], phone[i], product[i], cord[i])):		
		partition_product.append(total_top10_indicators_sense_line[i][0])
	else:
		partition_cord.append(total_top10_indicators_sense_line[i][0])

print(partition_cord, " is partition for cord")
print(partition_formation, " is partition for formation")
print(partition_division, " is partition for division")
print(partition_text, " is partition for text")
print(partition_phone, " is partition for phone")
print(partition_product, " is partition for product")

#For ambiguous word "serve"
SERVE6 = pmi(indicator_words_SERVE6, corpus_SERVE6,  total_top10_indicators_sense_serve, total_corpus_from_all_different_senses_SERVE)
SERVE10 = pmi(indicator_words_SERVE10, corpus_SERVE10, total_top10_indicators_sense_serve, total_corpus_from_all_different_senses_SERVE)
SERVE12 = pmi(indicator_words_SERVE12, corpus_SERVE12, total_top10_indicators_sense_serve, total_corpus_from_all_different_senses_SERVE)
SERVE2 = pmi(indicator_words_SERVE2, corpus_SERVE2, total_top10_indicators_sense_serve, total_corpus_from_all_different_senses_SERVE)
#Applying Flip-Flop
for i in range(0,len(total_top10_indicators_sense_serve)-1):
	if(SERVE6[i] == max(SERVE6[i], SERVE10[i], SERVE12[i], SERVE2[i])):
		partition_SERVE6.append(total_top10_indicators_sense_serve[i][0])
	elif(SERVE10[i] == max(SERVE6[i], SERVE10[i], SERVE12[i], SERVE2[i])):
		partition_SERVE10.append(total_top10_indicators_sense_serve[i][0])
	elif(SERVE12[i] == max(SERVE6[i], SERVE10[i], SERVE12[i], SERVE2[i])):	
		partition_SERVE12.append(total_top10_indicators_sense_serve[i][0])	
	else:
		partition_SERVE2.append(total_top10_indicators_sense_serve[i][0])

print(partition_SERVE2, " is partition for SERVE2")
print(partition_SERVE6, " is partition for SERVE6")
print(partition_SERVE10, " is partition for SERVE10")
print(partition_SERVE12, " is partition for SERVE12")

#For ambiguous word "interest"
interest_6 = pmi(indicator_words_interest_6, corpus_interest_6, total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
interest_1 = pmi(indicator_words_interest_1, corpus_interest_1,  total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
interest_4 = pmi(indicator_words_interest_4, corpus_interest_4,  total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
interest_3 = pmi(indicator_words_interest_3, corpus_interest_3,  total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
interest_2 = pmi(indicator_words_interest_2, corpus_interest_2,  total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
interest_5 = pmi(indicator_words_interest_5, corpus_interest_5,  total_top10_indicators_sense_interest, total_corpus_from_all_different_senses_INTEREST)
#Applying Flip-Flop
for i in range(0,len(total_top10_indicators_sense_interest)):
	if(interest_6[i] == max(interest_6[i], interest_1[i], interest_4[i], interest_3[i], interest_2[i], interest_5[i])):
		partition_interest_6.append(total_top10_indicators_sense_interest[i][0])
	elif(interest_5[i] == max(interest_6[i], interest_1[i], interest_4[i], interest_3[i], interest_2[i], interest_5[i])):
		partition_interest_5.append(total_top10_indicators_sense_interest[i][0])
	elif(interest_1[i] == max(interest_6[i], interest_1[i], interest_4[i], interest_3[i], interest_2[i], interest_5[i])):	
		partition_interest_1.append(total_top10_indicators_sense_interest[i][0])	
	elif(interest_4[i]== max(interest_6[i], interest_1[i], interest_4[i], interest_3[i], interest_2[i], interest_5[i])):	
		partition_interest_4.append(total_top10_indicators_sense_interest[i][0])
	elif(interest_3[i] == max(interest_6[i], interest_1[i], interest_4[i], interest_3[i], interest_2[i], interest_5[i])):		
		partition_interest_3.append(total_top10_indicators_sense_interest[i][0])
	else:
		partition_interest_2.append(total_top10_indicators_sense_interest[i][0])
print(partition_interest_5, " is partition for interest_5")
print(partition_interest_6, " is partition for interest_6")
print(partition_interest_1, " is partition for interest_1")
print(partition_interest_4, " is partition for interest_4")
print(partition_interest_3, " is partition for interest_3")
print(partition_interest_2, " is partition for interest_2")
