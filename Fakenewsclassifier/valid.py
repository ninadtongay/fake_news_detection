import tensorflow as tf
import numpy as np
import training
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
model=tf.keras.models.load_model('model.h5')
ps = PorterStemmer()
voc_size=5000
def data_preprocessing1(text_message):
      corpus = []
      review = re.sub('[^a-zA-Z]', ' ', text_message)
      review = review.lower()
      review = review.split()      
      review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
      review = ' '.join(review)
      corpus.append(review)
          
      return [one_hot(words,voc_size)for words in corpus] 

sentence_length=450
def embedding_representation(onehot_repr):
    
    embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sentence_length)
    return embedded_docs

