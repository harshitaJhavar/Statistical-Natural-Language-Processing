from nltk.tokenize import word_tokenize
import string
from collections import defaultdict
from matplotlib import pyplot as plt
from collections import Counter

def count_ngrams_hierarchical(text, n):
    counts={}
    for j in range(n):
        counts[j+1]=Counter(zip(*[text[i:] for i in range(j+1)]))
    return counts 

def compute_perplexity(jointProbabilityTest,distributionModel):
    from math import log
    base = 2
    loglikelihood = -sum(testNgramCount*log(distributionModel[testNgram],base)
                        for testNgram,testNgramCount in jointProbabilityTest.items())

    perp = base**loglikelihood
    return perp

class discounting_model:
    def __init__(self, d, unigramCounts, bigramCounts):
        self._bigramCounts = bigramCounts
        self._unigramCounts = unigramCounts
        self._d = d
        trainTokens = set(bigram[-1] for bigram in bigramCounts.keys())
        self._V = len( unigramCounts)
        self._R = ()
        self.R = Counter(bigram[0:-1] for bigram,count in bigramCounts.items() if count > 0)    # solution

    def prune(self, epsilon):
        unigrams= self._unigramCounts
        bigrams = self._bigramCounts
        N = sum(self._unigramCounts.values())

        pruned_unigrams = dict((k, v) for k, v in unigrams.items() if self.__unigram__(k) > epsilon)
        pruned_bigrams = dict((k, v) for k, v in bigrams.items() if self[k] > epsilon)

        new_model = discounting_model(self._d, pruned_unigrams, pruned_bigrams)
        return new_model

    def parameters(self):
        return len(self._bigramCounts) + len(self._unigramCounts)

    def __unigram__(self, unigram):
        unigramCount=self._unigramCounts.get(unigram, 0)
        V, d = self._V, self._d
        zeroGramProb=1/V

        N = sum(self._unigramCounts.values())
        return max(unigramCount-d,0) / N + d*V / N * zeroGramProb


    def __getitem__(self, bigram):
        bigramCount = self._bigramCounts.get(bigram, 0)
        history = bigram[0:1]
        historyCount = self._unigramCounts.get(history, 0)
        unigram = bigram[1:]
        
        V, d = self._V, self._d
        N = sum(self._unigramCounts.values())
        nPlus = self.R[history]

        
        uniGramProb = self.__unigram__(unigram)

        if historyCount>0:
            prob = max(bigramCount - d,0)/ historyCount + d*nPlus/historyCount * uniGramProb

        else:
            prob = uniGramProb

        return prob


if __name__ == "__main__":
    tokens = open('twain/pg119.txt', 'r', encoding='utf8').read().split()
    ngramCounts = count_ngrams_hierarchical(tokens, 2)

    tokens2 = open('twain/pg3176.txt', 'r', encoding='utf8').read().split() #tokenize('twain/pg3176.txt')
    ngramCountsTest = count_ngrams_hierarchical(tokens2, 2)
    N = sum(ngramCountsTest[1].values())-1

    mdl = discounting_model(0.9, ngramCounts[1], ngramCounts[2])

    jointDistribution= {bigram : bigramCount/N for bigram,bigramCount in ngramCountsTest[2].items()}
    ppl = compute_perplexity(jointDistribution, mdl)
    print("original PPL", ppl, mdl.parameters())

    for i in range(3, 7):
        pruned = mdl.prune(10**(-i))
        ppl2 = compute_perplexity(jointDistribution, pruned)
        print("PPL for 10^-", i, "is", ppl2, "with parameters", pruned.parameters())