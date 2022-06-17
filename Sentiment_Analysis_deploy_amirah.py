# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:27:43 2022

@author: Amirah Heng
"""


#%% Deployment unusually done on another PC/mobile phone
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# 1. Trained Model --> loading from .h5
# 2. tokenizer --> loading from json
# 3. MMS/OHE --> loading from pickle



TOKEN_PATH = os.path.join (os.getcwd(), 'tokenizer_sentiment.json')



#To load trained model
loaded_model = load_model(os.path.join(os.getcwd(),'friend_folder','model.h5'))
loaded_model.summary()
#To load tokenizer
with open(TOKEN_PATH, 'r') as json_file:
    loaded_token = json.load(json_file)
    

import pickle
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH,'wb') as file:
    loaded_ohe = pickle.load(file)




#%%
input_review = "The movie is so good. The trailer intrigues me to watch"

#preprocessing

input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ',input_review).lower().split()



tokenizer = tokenizer_from_json(loaded_token)
input_review_encoded = tokenizer.texts_to_sequences(input_review)


input_review_encoded= pad_sequences(np.array(input_review_encoded).T,
                                    maxlen=180, padding ='post', truncating ='post')

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis =-1))


with open(OHE_PATH,'rb') as file:
   loaded_ohe = pickle.load(file)

print(loaded_ohe.inverse_transform(outcome))#positive 
                #so the model has determined that input_review is positive
                # So the model is doing a good job here