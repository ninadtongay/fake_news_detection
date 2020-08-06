from flask import Flask,render_template,request
import tensorflow as tf
import numpy as np
import training
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import valid
import re
app = Flask(__name__)
mod=tf.keras.models.load_model('model.h5')
#for single string
# voc_size=5000
# ps = PorterStemmer()
# def data_preprocessing1(text_message): 
#     corpus = []
#     review = re.sub('[^a-zA-Z]', ' ', text_message)
#     review = review.lower()
#     review = review.split()      
#     review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
#     review = ' '.join(review)
#     corpus.append(review)
          
#     return [one_hot(words,voc_size)for words in corpus] 

# sentence_length=450
# def embedding_representation(onehot_repr):
    
#     embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sentence_length)
#     return embedded_docs


@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        news=myDict['test_result1']

        

        test_result1=news
        test_onehot_repr=valid.data_preprocessing1(test_result1)



        test_embedded_docs=valid.embedding_representation(test_onehot_repr)


        test_final=np.array(test_embedded_docs)
        model_prediction = mod.predict_classes(test_final)[0][0]
        model_prob=mod.predict_proba(test_final)[0][0]



        result=model_prediction

    
        
        print(result)
        if(result==0):
            value="False news detected"
        else:
            value="True news detected"
        print(model_prob)
        return render_template('show.html',inf=value)
    return render_template("index.html")
    # return 'News is :'+value


if __name__=="__main__":
    app.run(debug=True)