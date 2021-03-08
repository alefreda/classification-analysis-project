from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score,f1_score
#https://stackoverflow.com/questions/49785904/how-to-set-threshold-to-scikit-learn-random-forest-model

def scores(y_actual, yPredicted, thresh):
    
    auc = roc_auc_score(y_actual, yPredicted)
    accuracy = accuracy_score(y_actual, (yPredicted > thresh))
    recall = recall_score(y_actual, (yPredicted > thresh))
    precision = precision_score(y_actual, (yPredicted > thresh))
    fscore = f1_score(y_actual,(yPredicted > thresh) )

    print(f'AUC: {auc}\naccuracy: {accuracy} \nrecall: {recall} \
         \nprecision: {precision} \nfscore: {fscore} \n')

    return auc, accuracy, recall, precision, fscore

