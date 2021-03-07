from speech_utils import recognizer_source, speech_to_text, find_microphone
import random
import json
import json
from model import Model
from nlp_utils import tokenization as tk
from nlp_utils import stemming as stem
from nlp_utils import bag_of_words as bow
from nlp_utils import pos_tagging as pos
from speech import text_speech as ts
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------INTENTION CLASSIFICATION MODEL---------------------------------
with open('data_classification.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data_classification"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Model(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#-------------------------------------------------------------------------------------------
#----------------------------UPDATE CLASSIFICATION MODEL------------------------------------
with open('data_update.json', 'r') as json_data:
    intents_U = json.load(json_data)

FILE = "data_update"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words_U = data['all_words']
tags_U = data['tags']
model_state = data["model_state"]

model_update = Model(input_size, hidden_size, output_size).to(device)
model_update.load_state_dict(model_state)
model_update.eval()

#------------------------------------------------------------------------------------------
#----------------------------CHECK CLASSIFICATION MODEL------------------------------------
with open('data_check.json', 'r') as json_data:
    intents_C = json.load(json_data)

FILE = "data_check"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words_C = data['all_words']
tags_C = data['tags']
model_state = data["model_state"]

model_check = Model(input_size, hidden_size, output_size).to(device)
model_check.load_state_dict(model_state)
model_check.eval()

#------------------------------------------------------------------------------------------
#----------------------------UPDATE CLASSIFICATION MODEL------------------------------------
with open('initiative.json', 'r') as json_data:
    intents_initiative = json.load(json_data)


#------------------------------------------------------------------------------------------
#----------------------------MICROPHONE CHOICE---------------------------------------------
find_microphone()
print("type the Index of your microphone")
index=int(input())

r,source= recognizer_source(index)
print("'quit' to exit")
#------------------------------------------------------------------------------------------

'''
CHAT-LOOP
'''

user_name="Francesco"
bot_name="Gardener"

while True:


    
    

    #----------------------------USER AUDIO SPEECH TO TEXT-----------------------------------------------------------------
    print("________________________________________________________________________________________________________")
    sentence = speech_to_text(r,source)
    print(f"{user_name}: {sentence}")
    sentence = tk(sentence)

    #---------------------------------------------------------------------------------------------------------------- 
    #----------------------------FROM TOKENIZATION TO BAG OF WORRS AND NUMPY DATA------------------------------------
   
    #----DATA FOR THE FIST MODEL    
    X = bow(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)


    #----DATA FOR THE CHECK MODEL
    X_C = bow(sentence, all_words_C)
    X_C = X_C.reshape(1, X_C.shape[0])
    X_C = torch.from_numpy(X_C).to(device)


    #----DATA FOR THE UPDATE MODEL
    X_U = bow(sentence, all_words_U)
    X_U = X_U.reshape(1, X_U.shape[0])
    X_U = torch.from_numpy(X_U).to(device)


    #----CONDIZIONE PER USCIRE 
    if sentence == "quit":
        break
    #----------------------------------------------------------------------------------------------------------------
    


    

    # X data trought the fist model
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    
    
    
    
    text_miss= 'I do not understand, please repeate' 

    if prob.item() > 0.70:

        #CONTROLLO SE È CHECK O UPDATE o ALTRO
        
        for intent in intents['intents']:
            
            if tag == intent["tag"]:



                if tag == "CHECK":
                 #----------------------CONTROLLO TRA CHECK CHE TIPO DI LETTURA VUOLE FARE------------------
                 #X_C data trought the second model
                 output_C = model_check(X_C)
                 _, predicted_C = torch.max(output_C, dim=1)
                 tag_C = tags_C[predicted_C.item()]
                 probs_C = torch.softmax(output_C, dim=1)
                 prob_C = probs_C[0][predicted_C.item()]
                 
                 if prob_C.item() > 0.70:
                     for intent_C in intents_C['intents']:
                         if tag_C==intent_C["tag"]:
                             text=random.choice(intent_C['responses'])
                             print(f"{bot_name}: {text}")
                             ts(text)
                 else:                
                    print(f"{bot_name}: {text_miss}")
                    ts(text_miss)
                 #----------------------------------------
                 



                elif tag == "UPDATE":
                 #---------------------CONTROLLO TRA GLI UPDATE CHE TIPO DI AGGIORNAMENTO VUOLE FARE-------------------
                 #X_U data trought the third model
                 output_U = model_update(X_U)
                 _, predicted_U = torch.max(output_U, dim=1)
                 tag_U = tags_U[predicted_U.item()]
                 probs_U = torch.softmax(output_U, dim=1)
                 prob_U = probs_U[0][predicted_U.item()]
                 
                 if prob_U.item() > 0.70:
                     for intent_U in intents_U['intents']:
                         if tag_U==intent_U["tag"]:
                             text=random.choice(intent_U['responses'])
                             print(f"{bot_name}: {text}")
                             ts(text)                
                 else:
                    print(f"{bot_name}: {text_miss}")
                    ts(text_miss)
                 #----------------------------------------
                 


              
                else: 
                 #if is not UPDATE or CHECK intention is one others intentes and a random responce is choosen  
                 text=random.choice(intent['responses']) 
                 print(f"{bot_name}: {text}")
                 ts(text)



                #possibile presa di iniziatiza da parte del bot
                import random
                initiative=random.uniform(0, 1)
                print(initiative)
                for intent_initiative in intents_initiative['intents']:
                    if initiative >= 0.60:
                      print('here')
                      text=random.choice(intent_initiative['responses']) 
                      print(f"{bot_name}: {text}")
                      ts(text)

    else:
        print(f"{bot_name}: {text_miss}")
        ts(text_miss)


    initiative=random.uniform(0, 1)
    print(initiative)

    for intent_initiative in intents_initiative['intents']:
        if initiative >= 0.70:
            print('here')
            text=random.choice(intent_initiative['responses']) 
            print(f"{bot_name}: {text}")
            ts(text)    


