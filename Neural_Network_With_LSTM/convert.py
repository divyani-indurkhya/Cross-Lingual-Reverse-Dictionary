from flask import Flask, redirect, url_for, request
from collections import Counter, OrderedDict,defaultdict
import numpy as np
import string
import re
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os
from flask import render_template
      

app = Flask(__name__)


def data_processing(from_lang):

    punc_table = str.maketrans({key: None for key in string.punctuation})
    sentences = []
    targets = []
    count =0    
    with open('dataset/'+str(from_lang)+'_word_def.txt', 'r') as filee:     
        for i, line in enumerate(filee):
            words = line.strip('\n').split('\t')
            word = words[0]
            definitions = words[1].split(';')
            for definition in definitions:
                definition = re.sub("[\(\[].*?[\)\]]", "", definition).replace('  ', ' ')
                temp_word_list = definition.translate(punc_table).lower().split(' ')
                temp_word_list = list(filter(None, temp_word_list))
                mid_sent=['<start>'] + temp_word_list + ['<end>']
                if mid_sent in sentences:
                    continue
                sentences.append(['<start>'] + temp_word_list + ['<end>'])
                targets.append(word)
    WORD=[]
    for word_sublist in sentences:
        for word in word_sublist:
            WORD.append(word)
    words=WORD
    inf = float('inf')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    frequency_dict = OrderedDict({'<end>': inf, '<start>': inf})
    words_frequency_dict = sorted(Counter(words).most_common(None), key=lambda x:x[1], reverse=True)
    defs_frequency_dict = sorted(Counter(targets).most_common(None), key=lambda x:x[1], reverse=True)
    frequency_dict.update(words_frequency_dict)
    frequency_dict.update(defs_frequency_dict)
    frequency_dict.move_to_end('<start>', last=False)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    word2idx = OrderedDict([(item[0], i) for i,item in enumerate(frequency_dict.items())])
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))    
    return word2idx, idx2word

def word_mapping(word_line):
    word_line=str(word_line)
    word_line=word_line[1:len(word_line)-1]
    word_line=word_line.split(',')
    word_return=[]
    for word in word_line:
        word=word.strip()
        word=word[1:len(word)-1]
        word_return.append(word)
    return word_return

@app.route("/convert", methods = ['POST', 'GET'])
def index():
      if(request.method == 'POST'):
            from_lang = request.form['from_lang']
            to_lang = request.form['to_lang']
            definition = request.form['def']
            mapping_path='dataset/Cross_word_mapping/'
            word2idx, idx2word = data_processing(from_lang)
            model = load_model('model/'+from_lang+'.h5')
            data=pd.read_csv("data_lang.csv")
            data_len=len(data)
            path="dataset/Cross_word_mapping/"
            english_hindi=defaultdict(list)
            hindi_english=defaultdict(list)
            english_marathi=defaultdict(list)
            marathi_english=defaultdict(list)
            english_punjabi=defaultdict(list)
            punjabi_english=defaultdict(list)
            english_bengali=defaultdict(list)
            bengali_english=defaultdict(list)
            hindi_marathi=defaultdict(list)
            marathi_hindi=defaultdict(list)
            hindi_punjabi=defaultdict(list)
            punjabi_hindi=defaultdict(list)
            hindi_bengali=defaultdict(list)
            bengali_hindi=defaultdict(list)
            marathi_punjabi=defaultdict(list)
            punjabi_marathi=defaultdict(list)
            marathi_bengali=defaultdict(list)
            bengali_marathi=defaultdict(list)
            punjabi_bengali=defaultdict(list)
            bengali_punjabi=defaultdict(list)
            df=data[['english','hindi','marathi','punjabi','bengali']]
            df=np.array(df)#preparation for word mapping mapping
            for i in range(0,len(df)):
                  english=word_mapping(df[i][0])
                  hindi=word_mapping(df[i][1])
                  punjabi=word_mapping(df[i][3])
                  marathi=word_mapping(df[i][2])
                  bengali=word_mapping(df[i][4])
                  for word in english:
                        english_hindi[word].append(hindi)
                        english_punjabi[word].append(punjabi)
                        english_marathi[word].append(marathi)
                        english_bengali[word].append(bengali)
                  for word in hindi:
                        hindi_english[word].append(english)
                        hindi_punjabi[word].append(punjabi)
                        hindi_marathi[word].append(marathi)
                        hindi_bengali[word].append(bengali)
                  for word in punjabi:
                        punjabi_hindi[word].append(hindi)
                        punjabi_english[word].append(english)
                        punjabi_marathi[word].append(marathi)
                        punjabi_bengali[word].append(bengali)
                  for word in marathi:
                        marathi_hindi[word].append(hindi)
                        marathi_punjabi[word].append(punjabi)
                        marathi_english[word].append(english)
                        marathi_bengali[word].append(bengali)
                  for word in bengali:
                        bengali_hindi[word].append(hindi)
                        bengali_punjabi[word].append(punjabi)
                        bengali_marathi[word].append(marathi)
                        bengali_english[word].append(english)
            crosslist1=['english','hindi','marathi','punjabi','bengali']
            crosslist2=['english','hindi','marathi','punjabi','bengali']
            listofdict=[english_hindi,english_punjabi,english_marathi,english_bengali,hindi_english,hindi_punjabi,hindi_marathi            ,hindi_bengali,punjabi_hindi,punjabi_english,punjabi_marathi,punjabi_bengali,marathi_hindi,            marathi_punjabi,marathi_english,marathi_bengali,bengali_hindi,bengali_punjabi,bengali_marathi,bengali_english]
            listofdictname={'english_hindi':0,'english_punjabi':1,'english_marathi':2,'english_bengali':3,'hindi_english':4,                'hindi_punjabi':5,'hindi_marathi':6,'hindi_bengali':7,'punjabi_hindi':8,'punjabi_english':9,                'punjabi_marathi':10,'punjabi_bengali':11,'marathi_hindi':12,'marathi_punjabi':13,'marathi_english':14,                'marathi_bengali':15,'bengali_hindi':16,'bengali_punjabi':17,'bengali_marathi':18,'bengali_english':19}
            words = definition.split(' ')
            idxs = []
            for word in words:
                  if(word in word2idx):
                        idxs.append(word2idx[word])
            idxs = np.array([0] + idxs + [1]).reshape((1,len(idxs) + 2))
            
            prediction = model.predict(idxs, verbose=0)

            index = np.argmax(prediction)
            meaning = idx2word[index]
            if(from_lang == to_lang):
                  result=[meaning]
                  return render_template('index.html', result = result,definition=definition)
            temp=listofdict[listofdictname[from_lang+'_'+to_lang]][meaning]
            print(temp)
            ans=[]
            for line in temp:
                  for word in line:
                        ans.append(word)
                        if(len(ans)>4):
                              break
                  if(len(ans)>4):
                        break

            result=ans
            # for x in ans:
            #       result=result+str(x)+", "
            return render_template('index.html', result = result,definition=definition)
      else:
            return render_template('index.html', result = [],definition="")


app.run(debug = True)










    


