import json 
from nlp_utils import tokenization as tk
from nlp_utils import stemming as stem
from nlp_utils import bag_of_words as bow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import Model
import numpy as np



'''
STUCTURE OF JSON DATA
{
    "intents": [

        
      {
        "tag": "...",
        "patterns": ["...","..."],
        "responses": ["...","..."]
      },



      {
        "tag": "...",
        "patterns": ["...","..."],
        "responses": ["...","..."]
      },



      {
        "tag": "...",
        "patterns": ["...","..."],
        "responses": ["...","..."]
      },



      {
        "tag": "...",
        "patterns": ["...","..."],
        "responses": ["...","..."]
      },

      .....
      .....
      .....
      .....
    ]
  }
'''



def train_model_over_json(name_json, file_name):
 parole=[]
 tags=[]
 link=[]




 with open(name_json, 'r') as file:
    content=json.load(file)



 for intent in content['intents']:
    
    tag=intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        #TOKENIZATION
        #t è un arrey, stiamo costruendo l array parole estendendolo
        t=tk(pattern)
        parole.extend(t)

        #link , crea una tupla dove i corrispondenti pattern tokenizzati(un array) sono legati allo specifico tag
        link.append((t,tag))


    


 #STEMMING       
 ignore_words = ['?', '.', '!']
 parole = [stem(t) for t in parole if t not in ignore_words]
 parole=sorted(set(parole))

 x_train=[]
 y_train=[]

 for(t,tag) in link:
    bag = bow(t,parole)
    x_train.append(bag)

    label=tags.index(tag)
    y_train.append(label)

 #training data
 x_train=np.array(x_train)
 y_train=np.array(y_train)


 #dataset

 class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.X_train=x_train
        self.Y_train=y_train



    def __getitem__(self,index):  
        return self.X_train[index], self.Y_train[index]


    def __len__(self):
        return self.n_samples  




 '''
 HYPERPARAMETERS & TRAINING
 '''
 batch_size = 8
 num_epochs = 1000
 learning_rate = 0.001

 #dimensione dei layer nasconsti della nostra semplice rete neurale
 hidden_size = 8

 #l'output ha dimensioni uguale al numero dei tags, le nostre classi
 #il  modello deve  capire a quale tag appartiene la frase inserita
 output_size = len(tags)

 #input ha la stessa dimensione del bag of words
 #tutti gli elementi di x_train hanno la stessa lunghezza
 input_size = len(x_train[0])


 dataset=ChatDataset()
 train_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 model=Model(input_size, hidden_size, output_size)

 # Loss and optimizer
 criterion = nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

 # Train the model
 for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


 print(f'final loss: {loss.item():.4f}')

 data = {
 "model_state": model.state_dict(),
 "input_size": input_size,
 "hidden_size": hidden_size,
 "output_size": output_size,
 "all_words": parole,
 "tags": tags
 }

 FILE = file_name
 torch.save(data, FILE)

 print(f'training complete. file saved to {FILE}')


    


