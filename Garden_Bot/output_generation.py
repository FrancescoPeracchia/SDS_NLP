import numpy as np
from nltk.corpus import wordnet 



synonyms=[]
for parole in wordnet.synsets('home'):
    for lemma in parole.lemmas():
        synonyms.append(lemma.name())

for parole in synonyms:
 for sinomimi in wordnet.synsets(parole):
     print(sinomimi)
     print(sinomimi.examples())

#use similiarities function to respect to these example with the examples in the training set for the responce, and use the most evauleted word(the one with the most similar example to substiture the original word with the new one)  

