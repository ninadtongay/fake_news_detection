# import pandas as pd 
# from sklearn.utils import shuffle
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# import pickle



# fake =pd.read_csv("Fake1.csv")
# true=pd.read_csv("True1.csv")
# fake.insert(4,"actual_result","0")
# true.insert(4,"actual_result","1")
# combine=pd.concat([true,fake])
# combine=shuffle(combine)

# combine.to_csv('shuffled_dataset.csv')
# df=pd.read_csv('shuffled_dataset.csv')

# #droping null values from our dataframe 
# df=df.dropna()

# X=df.drop('actual_result',axis=1)

# y=df['actual_result']
# voc_size=5000
# textmessage=X.copy()
# textmessage.reset_index(inplace=True)

# ps = PorterStemmer()

# def data_preprocessing(text_message):
#     corpus = []
#     for i in range(0, len(text_message)):
    
#         review = re.sub('[^a-zA-Z]', ' ', text_message['text'][i])
#         review = review.lower()
#         review = review.split()
    
#         review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#         review = ' '.join(review)
#         corpus.append(review)
        
#     return [one_hot(words,voc_size)for words in corpus] 


# #for single string
# ps = PorterStemmer()

# def data_preprocessing1(text_message):
#       corpus = []
#       review = re.sub('[^a-zA-Z]', ' ', text_message)
#       review = review.lower()
#       review = review.split()      
#       review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#       review = ' '.join(review)
#       corpus.append(review)
          
#       return [one_hot(words,voc_size)for words in corpus] 


# onehot_repr=data_preprocessing(textmessage)

# sentence_length=450
# def embedding_representation(onehot_repr):
    
#     embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sentence_length)
#     return embedded_docs

# embedded_docs=embedding_representation(onehot_repr)

# embedding_vector_features= 40


# model=Sequential()


# model.add(Embedding(voc_size,embedding_vector_features,input_length=sentence_length))
# model.add(Dropout(0.3))
# model.add(LSTM(100))
# model.add(Dropout(0.3))

# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print(model.summary())

# X_final =np.array(embedded_docs)
# y_final=np.array(y)

# X_train,X_test,y_train,y_test=train_test_split(X_final,y_final,test_size=0.33,random_state=42)
# model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
# y_pred=model.predict_classes(X_test)

# confusion_matrix(y_test,y_pred)

# accuracy_score(y_test,y_pred)
# model.save('model')


