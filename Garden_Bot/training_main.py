from train import train_model_over_json

'''
questa parte del codice è utlizzata per generare i tre modelli, che classificheranno prima le INTENZIONI(classi) del utente, che sono:

GOODBYE
THANKS
ACTION
MOVE
ON
OFF
CHECK
UPDATE

UPDATE & CHECK sono intenzioni speciali, qui è necessario interagire con un DB o in generale con la knowledge base del nostro agente
in caso di lettura:CHECK
nell'altro in scrittura: UPDATE


qui entra in gioco il secondo classificatore, se nella classificazione precendente il testo ottenuto dall'audio ricevuto in input è classificato come UPDATE
segue una seconda classificazione per capire cosa va aggiornato

UPDATE :

FLOWERING-UPDATE
HUMIDITY-UPDATE
TEMPERATURE-UPDATE
LAST-WATERING-UPDATE
NEXT-CROP-UPDATE,

se ricade sotto la classe CHECK significa che l'utente pricipalmente si sta preoccupando di ricevere informazioni contenute del DB,
le quali non devono essere modificate ma solo riportate all' utente

CHECK:

FLOWERING-CHECK
HUMIDITY-CHECK
TEMPERATURE-CHECK
LAST-WATERING-CHECK
NEXT-CROP-CHECK,

'''
train_model_over_json('data_classification.json','data_classification')
train_model_over_json('data_check.json','data_check')
train_model_over_json('data_update.json','data_update')