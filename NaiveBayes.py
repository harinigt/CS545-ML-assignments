 # -*- coding: utf-8 -*-

"""
Created on Sat May 26 18:19:58 2018

@author: Harini Gowdagere Tulasidas 
PSU ID: 950961342
@Course : CS 545- Machine Learning
Programming Assignment2: Gaussian Naïve Bayes and Logistic Regression to classify 
the Spambase data from the UCI ML repository
"""
from __future__ import division
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import recall_score as recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


""" This method splits the dataset such that the test and train data has 2300 instances each 
the there are 40% spam and 60% no spam instances."""

def test_train_split():
    filename = "/Users/harinirahul/Desktop/CS - 545 - ML/PA2/spambase/spambase.data" 
    dataset = np.array(pd.read_csv(filename))
    temp = np.split(dataset, np.where(np.diff(dataset[:,-1]))[0]+1)
    
    spam = temp[0] 
    no_spam = temp[1]
    np.random.shuffle(spam)
    np.random.shuffle(no_spam)
    spam_size = int((len(spam)/2))
    no_spam_size = int((len(no_spam)/2))

    train_data = np.concatenate((spam[: spam_size, :],no_spam[:no_spam_size,:]), axis =0) 
    test_data = np.concatenate((spam[spam_size: , :],no_spam[no_spam_size:,:]), axis =0)
    
    train_labels = train_data[:,-1]
    train_labels = train_labels.reshape((len(train_labels),1))
    
    test_labels = test_data[:,-1]
    test_labels = test_labels.reshape((len(test_labels),1))
    
    return train_data,train_labels,test_data,test_labels

"""This is a utility method that computes mean and standard deviation for the features.
It also replaces the Standard deviation with minimum value of 0.0001 when it is 0. This is done
to avoid the errors while computing the log """

def mean_and_sd(data):
    x_mean = np.array(np.mean(data , axis = 0))
    x_std = np.array(np.std(data , axis = 0))
    x_std[x_std == 0.0] = 0.0001
  
    return x_mean.reshape((len(x_mean),1)) ,x_std.reshape((len(x_std),1))

"""
This method is used to calculate the prior probabilities for both the spam and Non spam classes.
"""
def calculate_probabilities(dataset):
    no_spam_count = 0 
    spam_count = 0
    no_spam = []
    spam = []
    for row in dataset:
        if row[-1]==1:
            spam_count+=1
            spam.append(row)  
        else:
            no_spam_count+=1
            no_spam.append(row)
             
    no_spam_prior = float(no_spam_count/len(dataset))
    spam_prior = float(spam_count/len(dataset))         
    print("prior Probability of the spam class: " ,spam_prior , "\n Prior probability of Non Spam class :", no_spam_prior)
    log_spam_prior = np.log(spam_prior)      
    log_no_spam_prior = np.log(no_spam_prior)  
    
    spam = np.array(spam)
    no_spam = np.array(no_spam) 
        
    spam_x_mean , spam_x_std = mean_and_sd(spam[: , :57])
    no_spam_x_mean , no_spam_x_std = mean_and_sd(no_spam[:,:57])
    return log_spam_prior , log_no_spam_prior , spam_x_mean , spam_x_std,no_spam_x_mean , no_spam_x_std 


""" This method is used to compute the probabilities for the Gaussian Naive Bayes algorithm
and classifies the instance as spam and non spam """
def gaussian_naive_bayes_classifier(log_spam_prior , log_no_spam_prior , spam_x_mean , spam_x_std ,no_spam_x_mean , no_spam_x_std,row):
    
    p_xi_cj_spam=(1/(np.sqrt(2*np.pi) * spam_x_std))*np.exp((-1)* (((row-spam_x_mean)**2)/(2*(spam_x_std**2))))
    p_xi_cj_no_spam = (1/(np.sqrt(2*np.pi) * no_spam_x_std))*np.exp((-1)* (((row-no_spam_x_mean)**2)/(2*(no_spam_x_std**2))))
    """Normalizing the Gaussian Naive Bayes probablities """
    p_xi_cj_spam[p_xi_cj_spam == 0.0] = 0.0001
    p_xi_cj_no_spam[p_xi_cj_no_spam == 0.0] = 0.0001
    
    log_naive_spam = np.sum(np.log(p_xi_cj_spam)) 
    log_naive_no_spam = np.sum(np.log(p_xi_cj_no_spam))
    
    no_spam_val = log_naive_no_spam+log_no_spam_prior 
    spam_val =  log_naive_spam+log_spam_prior
    
    return np.argmax([no_spam_val ,spam_val ]) 

"""This method has the final predictions of the Gaussian Naive Bayes Classifier for the dataset.   """
def predict(train_data,test_data):
    
    log_spam_prior , log_no_spam_prior , spam_x_mean , spam_x_std ,no_spam_x_mean , no_spam_x_std = calculate_probabilities(train_data)
    predicted_output = []
    for row in test_data:
        row = row.reshape((len(row),1))
        predicted_output.append(gaussian_naive_bayes_classifier
                                (log_spam_prior , log_no_spam_prior ,
                                                                spam_x_mean , spam_x_std ,no_spam_x_mean , no_spam_x_std,row))
    return predicted_output

"""The main method gets the predictions of the classifier and computes the various metrics
such as recall , accuracy and precision and also computes the confusion matrix  """

def main():
    train_data,train_labels,test_data,test_labels = test_train_split()
    predicted_output = predict(train_data,test_data[:,:57])    
    print("confusion matrix : \n" ,cm(test_labels,predicted_output))
    print("Recall : ",recall(test_labels,predicted_output))
    print("Accuracy:"  , accuracy_score(test_labels,predicted_output)*100 , "%" )
    print("precision : ",precision_score(test_labels,predicted_output))
    
    
if __name__== "__main__":
    main()
        
    

 
    
    


