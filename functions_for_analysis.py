import numpy as np
import pandas as pd
import math
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,  plot_confusion_matrix, classification_report, roc_auc_score, roc_curve

"""
#################################
###-----Data imputation----- ####
#################################
"""
#function to choose the number of k for knn imputation
def choose_k_for_knn_imputation(X, y):
    # evaluate each strategy on the dataset
    results = list()
    strategies = [str(i) for i in [1,3,5,7,9,15,18,21]] 
    for s in strategies:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('scale', StandardScaler()),('i', KNNImputer(n_neighbors=int(s))), ('m', LogisticRegression())])
        # evaluate the model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        # store results
        results.append(scores)
        print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=strategies, showmeans=True)
    pyplot.show()

    
#imputation using knn
def imputation_with_knn(data, nb):
    #--define imputer
    imputer = KNNImputer(n_neighbors=nb)
    #--fit on the dataset
    imputer.fit(data)
    #--transform the dataset
    imputed_data = imputer.transform(data)
    #--print total missing
    print('Missing values in the imputed dataset: %d' % sum(np.isnan(imputed_data).flatten()))
    #--transform to dataframe
    imputed_data = pd.DataFrame(imputed_data, columns = data.columns)
    return imputed_data


"""
#################################
###-----Data splitting----- ####
#################################
"""
#--split into training and test set
def train_split_test(input_data, y_data, indata_labels):
    X_train, X_test, y_train, y_test = train_test_split(
           input_data, y_data, test_size=0.3,
           stratify=y_data, random_state=0
    )
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    #transform the Target data into 'int'/binary
    y_train = y_train.astype("int")
    y_test = y_test.astype("int")
    #from numpy to dataframe
    X_train = pd.DataFrame(X_train, columns =indata_labels)
    X_test = pd.DataFrame(X_test, columns =indata_labels)
    
    return X_train, X_test, y_train, y_test

"""
#########################################
###-----Evaluation of ML models----- ####
#########################################
"""
#function to compute the evaluation metrics as: accuracy, precision, recall, etc.
def Print_confusion_matrix(cm, auc, heading): #cm for confusion matrix, heading is the title.
    print('\n', heading)
    true_negative  = cm[0,0]
    true_positive  = cm[1,1]
    false_negative = cm[1,0]
    false_positive = cm[0,1]
    total          = true_negative + true_positive + false_negative + false_positive
    accuracy       = (true_positive + true_negative)/total 
    precision      = (true_positive)/(true_positive + false_positive)
    recall         = (true_positive)/(true_positive + false_negative)
    misclassification_rate = (false_positive + false_negative)/total
    F1             = (2*true_positive)/(2*true_positive + false_positive + false_negative)
    mcc            = (true_positive*true_negative - false_positive*false_negative) / math.sqrt((true_positive+false_positive)*(true_positive+false_negative)*(true_negative+false_positive)*(true_negative+false_negative))
    print('accuracy.................%7.4f' % accuracy)
    print('precision................%7.4f' % precision)
    print('recall...................%7.4f' % recall)
    print('F1.......................%7.4f' % F1)
    print('auc......................%7.4f' % auc)
    print('mcc......................%7.4f' % mcc) 
    
    
#function to plot the ROC curve   
def plotROC(tpr, fpr, roc_auc, dataset, label=''):
    """
    Plot ROC curve from tpr and fpr.
    """  
    lw = 2
    plt.figure(figsize=(5, 5)) 
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='%s ROC curve (area = %0.2f)' % (label,roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic of the %s set' % dataset)
    plt.legend(loc="lower right")
    plt.show()
    
#function to evaluate the model by plotting the confusion matrix, the classification report and ROC curve of the input dataset.    
def evaluate_your_model_metrics(best_model, X_train, y_train, dataset_label):
    y_predicted_train = best_model.predict(X_train)
    probabilities_train = best_model.predict_proba(X_train)
    plot_confusion_matrix(best_model, X_train, y_train)
    plt.show()
    # plot the confusion matrix of the training set
    cm_train = confusion_matrix(y_train, y_predicted_train)
    auc_train = roc_auc_score(y_train, y_predicted_train)
    Print_confusion_matrix(cm_train, auc_train, '***Evaluation of the %s dataset***\n' %dataset_label)
    #plot the ROC curve
    fpr_train, tpr_train, th_train = roc_curve(y_train, probabilities_train[:, 1])
    plotROC(tpr_train, fpr_train, auc_train, '%s' %dataset_label, label='')
    #print the classification report
    c_report = classification_report(y_train, y_predicted_train)
    print('\nClassification report:\n', c_report)