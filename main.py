from utils import *
from visualization import *

import numpy as np
import pandas as pd

# Visualization

import seaborn as sns
import matplotlib.pyplot as plt

#Resample
from sklearn.utils import resample

# Metrics
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as cm

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, cross_validate, RepeatedKFold





dataset = pd.read_csv("dataset_diabetes/diabetic_data.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe().T)

IDs_mapping = pd.read_csv("dataset_diabetes/IDs_mapping.csv")

#-------------
#Data cleaning and preprocessing
#-------------


#replace all ? with NaN
dataset.replace('?', np.nan , inplace=True)

#change the target <30 >30 with 1 and No with 0
dataset['readmitted'] = dataset['readmitted'].replace(['<30','>30','NO'],['1','0','0'])
#print(dataset['readmitted'])

#Visualization of the age distribution
showAgeDistribution(dataset)

#Visualization of the target distribution
showTarget(dataset)

#missing value 
print(dataset.isna())
print(dataset.isnull().sum())

#drop some column not informative
dataset.drop(['examide' , 'citoglipton', 'weight','encounter_id','patient_nbr','payer_code','medical_specialty'], axis=1, inplace=True)

#gender

dataset['gender'].replace('Unknown/Invalid', np.nan , inplace=True)

#print(dataset['gender'].isna().sum()) 
# 3 are unknown 
mode_gender = dataset['gender'].mode()  #female
dataset['gender'].fillna(mode_gender[0], inplace=True)

#Visualization of the gender distribution
showGender(dataset)

#numerical columns
numerical_columns = dataset.select_dtypes('int64')
print(f'numerical columns: {numerical_columns} ')
print(f'Lenght: {len(numerical_columns)}')
#boxplot
column_box_plot = numerical_columns.loc[:,'time_in_hospital': 'number_diagnoses'].columns
print(column_box_plot)

boxplot_data(dataset,column_box_plot )



#print(dataset.isnull().sum())
#get the mode of race
mode_race = dataset['race'].mode()
#print(mode_race)
dataset['race'].fillna(mode_race[0], inplace=True)
#check if fillna success
#print(dataset["race"].isnull().sum())


#visualize gender age race 
gender_age_race(dataset)


#check if diag_1, diag_2, diag_3 are numeric column
diag_1_isNumeric = pd.to_numeric(dataset['diag_1'], errors='coerce').notnull().all()
diag_2_isNumeric = pd.to_numeric(dataset['diag_2'], errors='coerce').notnull().all()
diag_3_isNumeric = pd.to_numeric(dataset['diag_3'], errors='coerce').notnull().all()
print(f'Is numeric? diag_1: {diag_1_isNumeric}, diag_2: {diag_2_isNumeric}, diag_3: {diag_3_isNumeric}')
#print(dataset.isnull().sum())

#transform nan value with a string of nan for the categorical transformation 
dataset['diag_1'].fillna('Nan', inplace=True)
dataset['diag_2'].fillna('Nan', inplace=True)
dataset['diag_3'].fillna('Nan', inplace=True)



#diagnosis based on the ICD-9 codes
#https://en.wikipedia.org/wiki/List_of_ICD-9_codes
#https://en.wikipedia.org/wiki/List_of_ICD-9_codes_E_and_V_codes:_external_causes_of_injury_and_supplemental_classification

dataset.loc[dataset['diag_1'].str.contains('V'), 'diag_1'] = 'Vcodes'
dataset.loc[dataset['diag_2'].str.contains('V'), 'diag_2'] = 'Vcodes'
dataset.loc[dataset['diag_3'].str.contains('V'), 'diag_3'] = 'Vcodes'

dataset.loc[dataset['diag_1'].str.contains('E'), 'diag_1'] = 'Accidents'
dataset.loc[dataset['diag_2'].str.contains('E'), 'diag_2'] = 'Accidents'
dataset.loc[dataset['diag_3'].str.contains('E'), 'diag_3'] = 'Accidents'

def icd_9_code(code):
    classification_diagnosis = ''
    numeric_code = float(code)
    if numeric_code > 1 and numeric_code <=  139 :
        classification_diagnosis = 'Infectious'
    elif numeric_code == 0:
        classification_diagnosis = 'Vcodes'
    elif numeric_code == 1:
        classification_diagnosis = 'Accidents'
    elif numeric_code >= 140 and numeric_code <=  239:
        classification_diagnosis = 'Neoplasm'
    elif numeric_code >= 240 and numeric_code <=  279:
        classification_diagnosis = 'Metabolic'
    elif numeric_code >= 280 and numeric_code <=  289:
        classification_diagnosis = 'Blood'
    elif numeric_code >= 290 and numeric_code <=  319:
        classification_diagnosis = 'Mental'
    elif numeric_code >= 320 and numeric_code <=  389:
        classification_diagnosis = 'Nervous'
    elif numeric_code >= 390 and numeric_code <=  459:
        classification_diagnosis = 'Circulatory'
    elif numeric_code >= 460 and numeric_code <=  519:
        classification_diagnosis = 'Respiratory'
    elif numeric_code >= 520 and numeric_code <=  579:
        classification_diagnosis = 'Digestive'
    elif numeric_code >= 580 and numeric_code <=  629:
        classification_diagnosis = 'Genitourinary'
    elif numeric_code >= 630 and numeric_code <=  679:
        classification_diagnosis = 'Pregnancy'
    elif numeric_code >= 680 and numeric_code <=  709:
        classification_diagnosis = 'Skin'
    elif numeric_code >= 710 and numeric_code <=  739:
        classification_diagnosis = 'Muscoloskeletal'
    elif numeric_code >= 740 and numeric_code <=  999:
        classification_diagnosis = 'Congenital'
    elif numeric_code >= 760 and numeric_code <=  779:
        classification_diagnosis = 'Perinatal'
    elif numeric_code >= 780 and numeric_code <=  999:
        classification_diagnosis = 'Injury'
    else:
        classification_diagnosis = 'NaN'
    
    return classification_diagnosis
    
#print(dataset['diag_1'].head(50))
# 0 = Vcodes
# 1 = Accidents
diagnosis = ['diag_1','diag_2','diag_3']


for diag in diagnosis:
    dataset[diag] = dataset[diag].replace(['Vcodes'],'0')
    dataset[diag] = dataset[diag].replace(['Accidents'],'1')
    dataset[diag] = dataset[diag].astype(str)
    #dataset.loc[dataset[diag].astype(float) <=  139, diag] = 'Infectious'
    dataset[diag] = dataset[diag].apply(icd_9_code)

# An A1C test is a blood test that reflects your average blood glucose levels over the past 3 months

dataset['A1Cresult'] = dataset['A1Cresult'].apply(lambda x : 7 if x == '>7' 
                                                         else (8 if  x == '>8'                                                        
                                                         else ( 5 if x == 'Norm'
                                                         else  0)))

#Visualization of the diagnosis distribution
showDiagnosis(dataset,diagnosis)



print(dataset.head())

columns_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id','metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
        'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']

dataset[columns_ids] = dataset[columns_ids].astype('str')
one_hot_data = pd.get_dummies(dataset, columns=columns_ids)

df = one_hot_data.copy()

label_enc = LabelEncoder()
df['age'] = label_enc.fit_transform(df['age'])


for col in diagnosis:
    df[col] = label_enc.fit_transform(df[col])


features_1he = ['change', 'diabetesMed', 'gender', 'A1Cresult', 'max_glu_serum', 'race' ]
df = pd.get_dummies(df, columns=features_1he)





#split in train and test
X = df.drop(columns="readmitted", axis=1)
Y = df['readmitted'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)


#UNDERSAMPLING

concatenated_data = pd.concat([X_train, y_train], axis=1)
#print(concatenated_data.head())
target_0= concatenated_data.loc[concatenated_data['readmitted'] == 0]
target_1= concatenated_data.loc[concatenated_data['readmitted'] == 1]
#lenght of readmitted dataset
number_target_1 = len(target_1)
target_0_undersample = resample(target_0, n_samples = number_target_1, replace = False ,random_state = 42)

balanced_data = pd.concat([target_0_undersample, target_1])

#showTarget(balanced_data)


#split the balanced data in train x_train and y_train
y_train = balanced_data['readmitted']
X_train = balanced_data.drop('readmitted', axis=1)



thresh = 0.5


#-------------
#LOGISTIC REGRESSION
#-------------

print("---------------------")
print("Logistic Regression")
print("---------------------")

log_model = LogisticRegression(solver = "liblinear",random_state = 42)
log_model.fit(X_train, y_train)



#prediction train
y_train_preds_logistic = log_model.predict_proba(X_train)[:,1]
#prediction test
y_test_preds = log_model.predict_proba(X_test)[:,1]




print('Training:')
logistic_regression_train_auc, logistic_regression_train_accuracy, logistic_regression_train_recall, \
    logistic_regression_train_precision, logistic_regression_train_fscore = scores(y_train,y_train_preds_logistic, thresh)


print('Test:')
logistic_regression_test_auc, logistic_regression_test_accuracy, logistic_regression_test_recall, \
    logistic_regression_test_precision, logistic_regression_test_fscore = scores(y_test,y_test_preds, thresh)


# confusion matrix logistic regression
predictions_log_reg = log_model.predict(X_train)
train_score = round(accuracy_score(y_train, predictions_log_reg), 3)
cm_train = cm(y_train, predictions_log_reg)

#test

predictions_log_reg_test = log_model.predict(X_test)
test_score = round(accuracy_score(y_test, predictions_log_reg_test), 3)
cm_test = cm(y_test, predictions_log_reg_test)
showConfusionMatrixTest(cm_train,train_score, cm_test, test_score, 'Logistic Regression')





print("---------------------")
print("MULTINOMIAL NB")
print("---------------------")



multinomial_nb = MultinomialNB()

multinomial_nb.fit(X_train, y_train)


y_train_preds = multinomial_nb.predict_proba(X_train)[:,1]
y_test_preds = multinomial_nb.predict_proba(X_test)[:,1]


print('Training:')
multinomial_nb_train_auc, multinomial_nb_train_accuracy, multinomial_nb_train_recall, \
    multinomial_nb_train_precision, multinomial_nb_train_fscore =scores(y_train,y_train_preds, thresh)


print('Test:')
multinomial_nb_test_auc, multinomial_nb_test_accuracy, multinomial_nb_test_recall, \
    multinomial_nb_test_precision, multinomial_nb_test_fscore =scores(y_test,y_test_preds, thresh)

predictions = multinomial_nb.predict(X_train)
train_score = round(accuracy_score(y_train, predictions), 3)
cm_train = cm(y_train, predictions)


predictions_multinomial_nb_test = multinomial_nb.predict(X_test)
test_score_multinomialnb = round(accuracy_score(y_test, predictions_multinomial_nb_test), 3)
cm_test = cm(y_test, predictions_multinomial_nb_test)
showConfusionMatrixTest(cm_train,train_score, cm_test, test_score_multinomialnb, 'Multinomial NB')




#-------------
#RANDOM FOREST
#-------------

print("---------------------")
print("RANDOM FOREST")
print("---------------------")

random_forest = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=300, max_depth=3)
random_forest.fit(X_train, y_train)

y_train_preds = random_forest.predict_proba(X_train)[:,1]
y_test_preds = random_forest.predict_proba(X_test)[:,1]



print('Training:')
random_forest_train_auc, random_forest_train_accuracy, random_forest_train_recall, \
    random_forest_train_precision, random_forest_train_fscore =scores(y_train,y_train_preds, thresh)
print('Test:')
random_forest_test_auc, random_forest_test_accuracy, random_forest_test_recall, \
    random_forest_test_precision, random_forest_test_fscore = scores(y_test,y_test_preds, thresh)

# Confusion Matrix

predictions = random_forest.predict(X_train)
train_score = round(accuracy_score(y_train, predictions), 3)
cm_train = cm(y_train, predictions)

predictions_random_forest_test = random_forest.predict(X_test)
test_score_random_forest = round(accuracy_score(y_test, predictions_random_forest_test), 3)
cm_test = cm(y_test, predictions_random_forest_test)
showConfusionMatrixTest(cm_train,train_score, cm_test, test_score_random_forest, 'Random Forest')

print("---------------------")
print("RANDOM FOREST TUNING HYPERPARAMETERS")
print("---------------------")



param_grid_random_forest = {'max_depth': [2,5,8,10],
                            'n_estimators': [50, 100, 200, 500, 1000],
                            'max_features': [3,5,8,10],
                            'min_samples_split': [2,5,10]}

r_forest = RandomForestClassifier()

grid_search = GridSearchCV(r_forest, param_grid_random_forest, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(grid_search.cv_results_['mean_test_score'])
print(grid_search.best_params_)


print("---------------------")
print("RANDOM FOREST TUNED")
print("---------------------")

random_forest = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=500, max_depth=10, min_samples_split=10, max_features=8)
random_forest.fit(X_train, y_train)



y_train_preds = random_forest.predict_proba(X_train)[:,1]
y_test_preds = random_forest.predict_proba(X_test)[:,1]



print('Training:')
random_forest_train_auc, random_forest_train_accuracy, random_forest_train_recall, \
    random_forest_train_precision, random_forest_train_fscore =scores(y_train,y_train_preds, thresh)
print('Test:')
random_forest_test_auc, random_forest_test_accuracy, random_forest_test_recall, \
    random_forest_test_precision, random_forest_test_fscore = scores(y_test,y_test_preds, thresh)

# Confusion Matrix

predictions_tuned = random_forest.predict(X_train)
train_score_tuned = round(accuracy_score(y_train, predictions_tuned), 3)
cm_train = cm(y_train, predictions_tuned)

predictions_tuned_random_forest_test = random_forest.predict(X_test)
test_score_random_forest = round(accuracy_score(y_test, predictions_tuned_random_forest_test), 3)
cm_test = cm(y_test, predictions_tuned_random_forest_test)

showConfusionMatrixTest(cm_train,train_score_tuned, cm_test, test_score_random_forest, 'Random Forest Tuned')






print("---------------------")
print("MLP")
print("---------------------")

mlp = MLPClassifier(activation='relu', random_state=42, max_iter=500)
mlp.fit(X_train, y_train)

y_train_preds_mlp = mlp.predict_proba(X_train)[:,1]
y_test_preds_mlp = mlp.predict_proba(X_test)[:,1]



print('Training:')
mlp_train_auc, mlp_train_accuracy, mlp_train_recall, \
    mlp_train_precision, mlp_train_fscore =scores(y_train,y_train_preds_mlp, thresh)
print('Test:')
mlp_test_auc, mlp_test_accuracy, mlp_test_recall, \
    mlp_test_precision, mlp_test_fscore = scores(y_test,y_test_preds_mlp, thresh)




# Confusion Matrix

predictions_mlp_train = mlp.predict(X_train)
train_score_mlp = round(accuracy_score(y_train, predictions_mlp_train), 3)
cm_train = cm(y_train, predictions_mlp_train)

predictions_mlp_test = random_forest.predict(X_test)
test_score_mlp = round(accuracy_score(y_test, predictions_mlp_test), 3)
cm_test = cm(y_test, predictions_mlp_test)

showConfusionMatrixTest(cm_train,train_score_mlp, cm_test, test_score_mlp, 'MLP')


print("---------------------")
print("MLP TUNING HYPERPARAMETERS")
print("---------------------")


param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['logistic','tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    #tasso di regolarizzazione
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

mlp = MLPClassifier(random_state=42, max_iter=500)
#grid_search troppo onerosa
#grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, n_jobs=-1)
random_search_mlp = RandomizedSearchCV(mlp,param_grid_mlp, cv=5, n_jobs=-1)
random_search_mlp.fit(X_train, y_train)

print(random_search_mlp.cv_results_['mean_test_score'])
print(random_search_mlp.best_params_)


print("---------------------")
print("MLP Tuned")
print("---------------------")

mlp_tuned = MLPClassifier(solver = 'sgd', learning_rate = 'adaptive', hidden_layer_sizes= (100,), alpha = 0.05 , activation='relu', random_state=42, max_iter=500)
mlp_tuned.fit(X_train, y_train)

y_train_preds_mlp_tuned = mlp_tuned.predict_proba(X_train)[:,1]
y_test_preds_mlp_tuned = mlp_tuned.predict_proba(X_test)[:,1]



print('Training:')
mlp_tuned_train_auc, mlp_tuned_train_accuracy, mlp_tuned_train_recall, \
    mlp_tuned_train_precision, mlp_tuned_train_fscore =scores(y_train,y_train_preds_mlp_tuned, thresh)
print('Test:')
mlp_tuned_test_auc, mlp_tuned_test_accuracy, mlp_tuned_test_recall, \
    mlp_tuned_test_precision, mlp_tuned_test_fscore = scores(y_test,y_test_preds_mlp_tuned, thresh)


# Confusion Matrix

predictions_mlp_tuned_train = mlp_tuned.predict(X_train)
train_score_mlp_tuned = round(accuracy_score(y_train, predictions_mlp_tuned_train), 3)
cm_train = cm(y_train, predictions_mlp_tuned_train)

predictions_mlp_tuned_test = random_forest.predict(X_test)
test_score_mlp_tuned = round(accuracy_score(y_test, predictions_mlp_tuned_test), 3)
cm_test = cm(y_test, predictions_mlp_tuned_test)

showConfusionMatrixTest(cm_train,train_score_mlp_tuned, cm_test, test_score_mlp_tuned, 'MLP Tuned')






