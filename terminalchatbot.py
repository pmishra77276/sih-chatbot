import random
import json
import pickle
import numpy as np
import nltk
# import streamlit as st
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words1.pkl', 'rb'))
classes = pickle.load(open('classes1.pkl', 'rb'))
model = load_model('test1.h5')
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
def predict_class (sentence):
    bow = bag_of_words (sentence)
    
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    # print(res)
    # result1=[[i, r] for i, r in enumerate(res)]
    # result1.sort(key=lambda x: x[1],reverse=True)
    # print(result1)
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

def main():
    while(True):
        message = input("Enter messagae :")

        ints = predict_class (message)
        print(ints)
        # for i in range(len(ints)):
        flag=0
        for i in range(len(ints)):
            if(float(ints[i]["probability"])>=0.5):
                flag=1
                res = get_response (ints, intents)
                print(res)
                break
        if(flag==0):
            print("For this query I don't know the answer please contact (033) 23576008.")
        
        if "bye" in message.lower():
            break
            

if __name__ == "__main__":
    main()
