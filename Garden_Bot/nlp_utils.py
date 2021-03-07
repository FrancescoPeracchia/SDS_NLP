
'''
importare la libreria
package con un Tokenizzatore già trainato
'''
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



'''
questo metodo restituisce la frase tokenizzata

'''
def tokenization(frase):

    return nltk.word_tokenize(frase) 



'''
PortStemmer è uno dei possibile Stemmer che questa libreria fornisce
'''
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def stemming(parola):

    return stemmer.stem(parola.lower())






'''
in questa parte....
INPUT: frase_tokenizzata:["The","book","is","on","the","table"],["book","hi,"house","the","cat"....tutte le parole..]
OUTPUT:bag_of_words=[1,1,0,0,1,1] "the" e "book causano un match"
'''
def bag_of_words(frase_tokenizzata, parole):
    
    frase_tokenizzata=[stemming(t) for t in frase_tokenizzata]
    bag=np.zeros(len(parole), dtype=np.float32)

    for idx, w in enumerate(parole):
        if w in frase_tokenizzata:
            bag[idx] = 1.0 
   

    return bag



def  pos_tagging(frase):
    frase_tokenizzata=nltk.word_tokenize(frase) 
    print(frase_tokenizzata) 
    for parole in frase_tokenizzata:
      tag=nltk.pos_tag(frase_tokenizzata)
    return tag




'''
#------------------------------TESTING TOKENIZATION-----------------------------------------------
t="The book is on the table"
print(t)
t=tokenization(t)
print(t)
#-------------------------------------------------------------------------------------------------
'''



'''
#------------------------------TESTING STEMMING---------------------------------------------------
t=["study","studing","studied","Studies"]
print(t)
stemmed_t=[stemming(w) for w in t ]
print(stemmed_t)
#-------------------------------------------------------------------------------------------------
'''



'''
#------------------------------TESTING BAG_OF_WORDS------------------------------------------------
frase_tokenizzata=["the","book","is","on","the","table"]
parole=["the","my","on","table","book","floor","house","you","i","gain","loose","italy","cat","dog","mum"]
bag=bag_of_words(frase_tokenizzata,parole)
print(bag)
#--------------------------------------------------------------------------------------------------
'''