# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:19:41 2022

@author: Amirah Heng
"""

import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional,Embedding
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer

CSV_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

#EDA
#STEP 1 - Data Loading

df = pd.read_csv(CSV_URL)

#
df_copy = df.copy()  #backup

# Let's say if we mess up the data right, 
# we can copy the df_copy again and again, so really a back up one ah! smart!
# df_copyII = df_copy.copy()

#%% STEP 2 - Data Inspection
df.head(10)
df.tail(10)

df.info()
df.describe().T #cant get anything much though due to its non-numeric


df['sentiment'].unique() #to get the unique target variables
                         #only got positive and negative sentiment
                         #Kalau use Deep Learning, need to do OHE for this

df['review'][0] #cant do much
                #but can do slicing to see
df['sentiment'][0] # 0 indicates positive reviews

df['sentiment'][1] #Negative sentiments
df['review'][1]    #Negative reviews
# Is there anything there to remove? Funny Characters! (HTML's ones), slashes
# So need to remove because later under tokenization, it will consider them as a char

df.isna().sum() # no missing values

df.duplicated().sum() #We have 418 duplicated datas
df[df.duplicated()] #Extracting the duplicated data 

## STUFFS TO REMOVE!
# <br /> HTMLS tags have to be removed    
# Numbers can be filtered
# need to remove duplicated datas


#%% STEP 3 - Data Cleaning

#3.1 Removing duplicated datas
df = df.drop_duplicates() # so now we have (49582,2) data from (50000,2)

#3.2 Removing HTML tags
# # Method 1
# '<br /> dhgsajklfgfdhsjka <br />'.replace('br />',' ') #replacing with nothing  
# # Method 2
# df['review'][0].replace('br />',' ') #no more br already now
#                                 # need to do for loop because we need to do for all row
# Method 3 
# Even faster way - remove whatever in <>
review = df['review'].values # Features: X #Extract the values to make review=sentiment len
sentiment = df['sentiment'] # Target: y


for index,rev in enumerate(review):
    # remove html tags
    # re.sub('<.*?>',rev) #? means dont be greedy
                        # so it wont convert anything anything beyond diamond bracket
                        # . means any characters
                        # * means zero or more occurences
                        # Any character except new lines (/n)
                        
    review[index] = re.sub('<.*?>',' ',rev) #'<.*?>' means we wanna remove all these
                            #' ' to replace with empty spac

                            # rev is our data
#3.3 Removing Numerics
    # convert into lower case
    # remove number
    # ^ means NOT alphabet
    review[index] = re.sub('[^a-zA-Z]',' ',rev).lower().split()
                                # substituting that is not a-z and A-Z 
                                # .. will be replaced with a space
                                # Hence, all numeric will be removed
                                # so now we have changed every word into lower 
                                # and splitted them into a list of words
                                
review[10] #so can see all the words has been split into a list of words
    
# 'ABC def hij'.lower().split()   # this is how we split each words
# 'I am a Data Scientist'.lower().split()   
    # Why need to use enumerate?
# but review and sentiment is not equal lor!    
# review has more data than sentiment
# extract the values.. 
# BUT WHY? because the datas are linked to df together

# watch the video and relate with the concept of a, b and pop


#%% STEP 4 - Features Selection

# Nothing to select since this NLP data

#%% STEP 5 - Preprocessing
#           1) Convert into lower case - done in Data Cleaning

#           2) Tokenization
# can never guess suitable amount of vocabulary size
# do scientific way ya
# np.unique() doesnt work on NLP like it would on numeric data
# temp = df['review'].unique() # see doesnt work!
#  BUT MR WARREN RUN SOMETHING WITH temp THAT IT SHOWED 8000 DATAS
vocab_size = 10000
oov_token = 'OOV'


tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review) #only the reviews have to be tokenized
                               # Sentiments --> OHE
word_index = tokenizer.word_index
print(word_index)            

# so need to encode all this into numbers to fit the review
train_sequences = tokenizer.texts_to_sequences(review)     
# so all the words now are in numerics              

#           3) Padding  & Truncating
# len(train_sequences[0])
# len(train_sequences[1])
# len(train_sequences[2])
# len(train_sequences[3])  # The number of words user has commented

# for padding we can choose either mean or median

length_of_review =[len(i) for i in train_sequences]

import numpy as np

# np.mean(length_of_review) # to get the number of mean words => 238 words
np.median(length_of_review) # to get the median words => 178 words
# can use the size of the train_sequences words as well

# pick the reasonable padding value
# we are choosing median for our padding values
# Padding is to make each length to be similar
max_len = 180

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_review = pad_sequences(train_sequences,
                              maxlen=max_len,
                              padding='post',
                              truncating='post')
                                # so now all in equal length already now
                                # 1 is OOV 

# 4) One Hot Encoding for the Target - Sentiment

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

#           5) Train-test-split because this is a classification problem

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)
X_train= np.expand_dims(X_train, axis=-1)
X_test= np.expand_dims(X_test, axis=-1)

#%% Model Development

# USE LSTM layers, dropout, dense, input
# acheive > 90% F1 score


num_node=128
drop_rate=0.2
ouput_node=2 
embedding_dim = 64

model = Sequential()
model.add(Input(shape=180))
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=True))) #214k become 400k double the
# model.add(LSTM(num_node,return_sequences=True)) 
model.add(Dropout(drop_rate))
model.add(LSTM(num_node)) #going to lower dense layer no need return_sequences
model.add(Dropout(drop_rate))
model.add(Dense(num_node, activation ='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(ouput_node, activation ='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics ='acc')

from tensorflow.keras.utils import plot_model
plot_model(model)
hist= model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
#0.57% 1st train #afterbidirectional still not improve

hist_keys = [i for i in hist.history.keys()]

plt.figure()
plt.plot(hist.history[hist_keys[0]], 'r--', label='Training loss')
plt.plot(hist.history[hist_keys[2]], label='validation loss')
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history[hist_keys[1]],'r--', label='Training acc')
plt.plot(hist.history[hist_keys[3]], label='validation acc')
plt.legend()
plt.show()



#%% Model Evaluation


y_true = y_test
y_pred = model.predict(X_test)

y_true= np.argmax(y_true,axis=1)
y_pred= np.argmax(y_pred,axis=1)

#%%
print(classification_report(y_true, y_pred)) 
print(accuracy_score(y_true,y_pred)) #0.56
print(confusion_matrix(y_true, y_pred))

#%% Model Saving
import os
MODEL_SAVE = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE)

import json
token_json = tokenizer.to_json()
TOKEN_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')

with open(TOKEN_PATH, 'w') as file:
    json.dump(token_json, file)

import pickle
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe,file)



#%%Discussion /Report

# Discuss your results
# model achieved around 84% accuracy during training
#  Recall(Sensitivity) and f1-score  reported 87 and 84% respectively
# However the model starts to overfit after 2nd epoch
# EarlyStopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different DL architecture for example BERT model, transformer
# model, GPT3 model may help to improve the model

#1) Resuts ----> discussion on the results
#2) gives suggestion ---> how to improve your model
#3) gather evidences showing what went wrong during training/ model


# (df['review'] == 'positive').sum()  #Make correction on your own!


















