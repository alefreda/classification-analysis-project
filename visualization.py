
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report

# ---------------
#Trget visualization 
# ---------------

def showTarget(dataframe):
    """
    visualization distribution target
    """
    sns.countplot(x=dataframe['readmitted'], data=dataframe, palette="pastel", edgecolor=".3")
    plt.show()


# ---------------
#Gender visualization 
# ---------------
def showGender(dataframe):
    """
    visualization distribution target
    """
    sns.countplot(x=dataframe['gender'], data=dataframe, palette="pastel", edgecolor=".3")
    plt.show()





#----------------
#Age visualization
# ---------------
 
#histogram
def showAgeDistribution(dataframe):
    fig=plt.figure() 
    ax = fig.add_subplot(1,1,1)
    #Variable
    ax.hist(dataframe['age'],bins = 7) 
    plt.title('Age distribution')
    plt.xlabel('Age')
    plt.ylabel('#Patients')
    plt.show()

#boxplot
def ageBoxplot(dataframe):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    #   Variable
    ax.boxplot(dataframe['age'])
    plt.show()




#----------------
#Diagnosis visualization
# ---------------
def showDiagnosis(dataframe, listOfDiag):
    """
    docstring
    """
    fig, ax =plt.subplots(nrows=3,ncols=1,figsize=(15,12))
    count =0
    for i in listOfDiag:
        sns.countplot(x = dataframe[i], hue=dataframe['readmitted'], palette='YlOrBr', ax=ax[count]);
        count = count+1
    plt.show()


#----------------
#Boxplot numerical columns
# ---------------
def boxplot_data(dataframe,columns):
    count = 0
    fig, ax =plt.subplots(nrows=2,ncols=4, figsize=(16,8))
    for i in range(2):
        for j in range(4):
            sns.boxplot(x = dataframe[columns[count]], palette="rocket", ax=ax[i][j])  
            count = count+1
    
    plt.show()




def gender_age_race(dataframe):
    visual_list = ['gender','age','race']
    fig, ax =plt.subplots(nrows=1,ncols=3,figsize=(24,8))
    count =0
    for i in visual_list:
        sns.countplot(dataframe[i], hue=dataframe['readmitted'], palette='rocket', ax=ax[count]);
        count = count+1
    
    plt.show()


#----------------
#Confusion matrix
# ---------------

def showConfusionMatrix(cm_train, train_score, title):
    """
    visualization confusion matrix
    """
    fig, ax =plt.subplots(nrows=1,ncols=1,figsize=(10,8))
    sns.heatmap(cm_train, annot=True, fmt=".0f", cmap="Blues")
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Values')
    ax.set_title(f' {title} \n \nTest MLP Accuracy Score: {train_score}')
    plt.show()


def showConfusionMatrixTest(train_matrix, train_score, test_matrix, test_score, title):
    """
    visualization confusion matrix
    """
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5)) 
    sns.heatmap(train_matrix, annot=True, fmt=".0f",ax=ax1, cmap="Blues")
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Actual Values')
    ax1.set_title(f' {title} \n \nTrain Accuracy Score: {train_score}')
    sns.heatmap(test_matrix, annot=True, fmt=".0f", ax=ax2, cmap="Blues")
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Actual Values')
    ax2.set_title(f' {title} \n \nTest Accuracy Score: {test_score}')
    plt.show()


    
def roc_curve(clf, X_test, y_test):
    metrics.plot_roc_curve(clf, X_test, y_test)
    plt.show()