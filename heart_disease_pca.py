#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:04:52 2019

@author: ahmetkaanipekoren
"""



def data_preparation():
    df = pd.read_csv("heart.csv")

    return df


def normalization(df):
    
    df_norm = (df-df.min()) / (df.max() - df.min())
    
    return df_norm


def x_y_values(df):
    df_y = df.loc[:,"target"]
    df_x = df.drop(["target"],axis=1)
    
    
    return df_x,df_y

    
def train_test(df_x,df_y):
    
    train_x,test_x,train_y,test_y = train_test_split(df_x,df_y,test_size = 0.2,random_state=42)

    train_x,test_x,train_y,test_y = np.array(train_x),np.array(test_x),np.array(train_y),np.array(test_y)
 
    return train_x,test_x,train_y,test_y



def eig_pairs(eigen_value,eigen_vector):
    
    eigen_pairs = []
    for i in range(len(eigen_value)):
        eigen_pairs.append((np.abs(eigen_value[i]), eigen_vector[:,i]))

    eigen_pairs = sorted(eigen_pairs,key=lambda k: k[0], reverse=True)
    return eigen_pairs

    
def cumulative_variance(eigenvalues):
    
    sum_eigen_values = sum(eigenvalues)
    cum_var_list = []
    for i in eigenvalues:
        cum_var_list.append(i/sum_eigen_values)
        
    cum_var = np.cumsum(cum_var_list)
    
    #plt.plot(cum_var)
    #plt.grid(True)
    #plt.show()
    
    return cum_var


def classification(df_x,eigenpairs):
    
    reduced_matrix = np.hstack((eigenpairs[0][1].reshape(13,1),eigenpairs[1][1].reshape(13,1)))
    
    new_data_set = np.dot(df_x,reduced_matrix)

   # plt.plot(new_data_set,"ko")
   # plt.show()       
    
    return new_data_set
    
    
def drawing_new_points(df_y,data_set):
    
    sum_red = 0,0
    sum_green = 0,0
    count_red = 0
    count_green = 0
    for i in range(len(df_y)):
        if df_y[i] == 1 :
            plt.scatter(data_set.T[0][i],data_set.T[1][i], color="green")
            sum_red = sum_red + data_set[i]
            count_green += 1
        elif df_y[i] == 0:
            plt.scatter(data_set.T[0][i],data_set.T[1][i], color ="red", alpha = 0.2)
            sum_green = sum_green + data_set[i]
            count_red += 1
            
    
    plt.show()
    
    
    return sum_green / count_green , sum_red / count_red
    
    

def svm_classifier(x_train,y_train,x_test,y_test):
    from sklearn.svm import SVC
    
    svm = SVC(random_state = 1, degree=2)
    svm.fit(x_train,y_train)
    
    acc_svm = svm.score(x_test,y_test)
   

    
    return acc_svm




if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    df = data_preparation()
    df_norm = normalization(df)
    df_x,df_y = x_y_values(df_norm)
    
    cov_mat = np.cov(df_x.T)
    eigenvalues,eigenvectors = np.linalg.eig(cov_mat)
    eigen_pairs = eig_pairs(eigenvalues,eigenvectors)
    cum_var = cumulative_variance(eigenvalues)
    new_data_set = classification(df_x,eigen_pairs)
    mean_green, mean_red = drawing_new_points(df_y,new_data_set)
    
    train_x,test_x,train_y,test_y = train_test(new_data_set,df_y)
    acc_svm = svm_classifier(train_x,train_y,test_x,test_y)
    
    
   


