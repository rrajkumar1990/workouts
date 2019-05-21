# -*- coding: utf-8 -*-
"""
Created on Wed May 22 00:38:31 2019

@author: rrajkumar1990
"""

import nltk
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
from pathlib import Path
import random
from sklearn.linear_model import LogisticRegression



#helper fucntion checking ascii
def check_for_nonascii(text):
    '''helper fucntion to check the passed text doesnt contain non ascii'''
    return ''.join( word for word in text if ord(word) < 128)


#helper fucntion for loading data set
def data_load():
    negative_review = []
    positive_review =[]
    
    #loading path from my folder negative and postive folde paths
    
    path = Path().absolute()
    negative_review_strings = os.listdir(path+'\neg') # neg is the folder for negative review files
    positive_review_strings = os.listdir(path+'\pos') # pos is the folder for positive review files
    
    #fetching our text from the directory to our variables respectively    
    for negative_review_string in negative_review_strings :
        with open (path+'\\neg\\'+str(negative_review_string)) as f :
            negative_review.append(check_for_nonascii(f.read()))
        
    for positive_review_string in positive_review_strings :
        with open (path+'\\pos\\'+str(positive_review_string)) as f :
            positive_review.append(check_for_nonascii(f.read()))

    negative_dic = [ dict(text= i, label = 1) for i in negative_review ] # lets label 1 for negative
    positive_dic = [ dict(text= i, label = 0) for i in positive_review ]  # lets label 0 for Positive

    data = negative_dic + positive_dic

    #now our data variable has positive and negative appended but negative and postive are in order for our algorithm to learn properly we should shuffle the data        
    random.shuffle(data)  # random.shuffle function used for shuffling the list
    
    #Now lets segregate the data for raw features and lables
    return [i['text'] for i in data], [i['lable'] for i in data]  # first return variable is the shuffles lisr of raw varibals and the second variable is the corresponding labels


#lets invoke the data_load function

X, y = data_load()

#set our input local variables for vectorizing the raw text
minimum_word_frequency = 10 
maximum_word_frequency = 250
pattern =r'\w+' #we need all alphanumeric characters in the text for except the special characters 

#Our X is still raw text we need to convert them for our algorithm
#lets use the term frequency inverse frequecny vectorizer


vectorizer = TfidfVectorizer(min_df=minimum_word_frequency, max_df=maximum_word_frequency, stop_words="english", token_pattern=pattern)

X = vectorizer.fit_transform(X).todense() # we have succesfully fit the raw text to the TfidfVectorizer

#our data is ready for feeding to a model
#lets split the input data to tran and test split using sklearn - train_test_split
spli_size=0.30 # 30% of the data to be splitted for test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=spli_size, 
                                                    random_state=42)

#lets use the Logitic Regression algorithm for our solution
#feel free to use any algorithms

lr = LogisticRegression() #lets use the default params , we can try tuningthe parameters may be use l1 penatly but for now lets go ahead with default params inthat case its l2 penalty (Ridge)

lr.fit(X_train,y_train) 

#Now the model is fit lets predict !!!! :)
pred =lr.predict(X_test)


#hurray we have predicted lets see our accuracy score using from sklearn.metrics import accuracy_score 

score = accuracy_score(y_test,pred)

#from my given data i could get accuracy of 86% still it needs tuning but for the given data set and default params it seems good
#lets check our confusion matrix frm sklearn.metrics import confusion_matrix function

conf_mat = confusion_matrix(y_test,pred)
#result of conf_mat
#array([[238,  52],[ 37, 233]], dtype=int64)
#we can see the True negative and true positive seems to be good but we need to work on our false positive and false negative which can be improved by having a better tuned model
 



    
