import pandas as pd
import numpy as np
import gc
import os
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import re
import tensorflow as tf
from transformers import TFXLNetModel, XLNetTokenizer
from transformers import AutoTokenizer, AutoModel, TFAutoModel
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import ParameterSampler
import tensorflow_hub as hub
import json
import csv 

import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.keras
import tensorflow_text as text
import seaborn as sns
from sentence_transformers import SentenceTransformer

torch.cuda.empty_cache()

os.environ['DATABRICKS_USERNAME'] = 'karengar@stanford.edu' 
os.environ['DATABRICKS_PASSWORD'] = 'Projectcs224*'

data_test = pd.read_csv('/root/data/ed_test.csv')
data_train = pd.read_csv('/root/data/ed_train.csv')
data_val = pd.read_csv('/root/data/ed_val.csv')
data_val.head()

def plot_roc_curve(y_proba, y_test, labels, mlflow, split_type):
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    fig = plt.figure(figsize = (17, 8))
    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovr = {}
    for i in range(len(labels)):
        # Gets the class
        c = labels[i]

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop = True)

        # Plots the probability distribution for the class and the rest
        fig_upper = plt.subplot(2, 4, i+1)
        x = df_aux[df_aux['class'] ==0]
        plt.hist(x['prob'], density=True, label='Rest')
        y = df_aux[df_aux['class'] ==1]
        plt.hist(y['prob'], density=True, label=f" {c}", bins = bins)
        plt.title(c)
        plt.legend(loc='upper right')
        plt.xlabel(f"P(x = {c})")
        #plt.show()
        #mlflow.log_figure(fig, f"Histogram: {c}.png") 


        # Calculates the ROC Coordinates and plots the ROC Curves
        fpr, tpr, thresholds = roc_curve(df_aux['class'], df_aux['prob'], pos_label=1)
        mauc = auc(fpr, tpr)
        fig_bottom = plt.subplot(2, 4, i+5)
        plt.title( f"ROC: {c}")
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % mauc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        #plt.show()
        mlflow.log_metric("auc_" + split_type + "_"+  c, mauc) 

    plt.tight_layout()
    mlflow.log_figure(fig, "Hist_and_ROC_" + c+ "_"+ split_type + ".png") 
    
def evaluate(y_test, y_pred, mlflow,  split_type):
    """
    Evaluation function. For each of the text in evaluation data, it reads the score from
    the predictions made. And based on this, it calculates the values of
    True positive, True negative, False positive, and False negative.

    :param y_test: true labels
    :param y_pred: predicted labels
    :param labels: list of possible labels
    :return: evaluation metrics for classification like, precision, recall, and f_score
    """
     
    b = np.zeros_like(y_pred)
    b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
    y_proba = y_pred

    y_pred = pd.DataFrame(b, columns = list(y_test.columns))
    y_pred = y_pred.idxmax(axis=1)
    y_test = y_test.idxmax(axis=1)

    labels = list(y_test.unique())
    labels = sorted(labels)

    plot_roc_curve(y_proba, y_test, labels, mlflow, split_type)

   
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = matrix.diagonal()/matrix.sum(axis=1)
    for i, accu in enumerate(accuracy):
        mlflow.log_metric("accu_" + split_type + "_"+  labels[i] , accu) 

    # importing accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= None)
    recall = recall_score(y_test, y_pred, average= None)
    f1score = f1_score(y_test, y_pred, average= None)


    report = classification_report(y_test, y_pred, target_names = labels)
    print(report)
    f_name = 'report_ '+ split_type +'.yaml'
    mlflow.log_dict(report, f_name)
    #mlflow.log_metric("auc_" + split_type, mauc) 
    mlflow.log_metric("accuracy_"+ split_type, accuracy)
   

    return accuracy, precision, recall, f1score




def mlflow_log_parameters(parameter):
    # Log parameters
    mlflow.log_param("dropout", parameter['dropout'])
    mlflow.log_param("learning_rate", parameter['learning_rate'])
    mlflow.log_param("epochs", parameter['epochs'])
    mlflow.log_param("batch_size", parameter['batch_size'])



def create_bert(bert, learning_rate=1e-3, dropout=0.1):
    input_ids = tf.keras.layers.Input(shape=(158,), dtype=tf.int32, name="input_ids")
    input_mask = tf.keras.layers.Input(shape=(158,), dtype=tf.int32, name="attention_mask")

    embeddings = bert(input_ids, attention_mask = input_mask)[0]
    bert.trainable = False
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out.trainable = False
    y = tf.keras.layers.Dense(4, activation='softmax', name='outputs')(out)

    # Compile model
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', 
               metrics=[tf.keras.metrics.AUC(), 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model


def training_test_mae( mlflow, history):
    # Plot training and test loss at each epoch 
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training acc')
    #plt.plot(history.history['val_accuracy'], label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    mlflow.log_figure(fig, "training_validation_accuracy.png") 

    fig, ax = plt.subplots()
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    mlflow.log_figure(fig, "training_validation_loss.png") 
    
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/karengar@stanford.edu/Bio_BERT_l2v")

X_train = data_train.drop(['Unnamed: 0', 'ED_dispo', 'CSN', 'Rads_order_ID'], axis=1)
X_test =  data_test.drop(['Unnamed: 0', 'ED_dispo', 'CSN', 'Rads_order_ID'], axis=1)
X_val = data_val.drop(['Unnamed: 0', 'ED_dispo', 'CSN', 'Rads_order_ID'], axis=1)

# Assigning numerical values and storing in another column
y_train =  pd.get_dummies(data_train['ED_dispo'])
y_test = pd.get_dummies(data_test['ED_dispo'])[list(y_train.columns)]
y_val =  pd.get_dummies(data_val['ED_dispo'])[list(y_train.columns)]

del data_train, data_test, data_val 

rng = np.random.RandomState(0)
param_dist = {    'dropout': [0.1,0.2, 0.3],
                  'learning_rate': loguniform.rvs(1e-4, 1e-2, size= 10),
                  'epochs': [1,3,5],
                  'batch_size':[32, 64]
                  }
            
dict_parameters = ParameterSampler(param_distributions=param_dist, n_iter= 10, random_state=rng)

tokenizer = AutoTokenizer.from_pretrained("AdwayK/biobert_ncbi_disease_ner_tuned_on_TAC2017")
bert = TFAutoModel.from_pretrained("AdwayK/biobert_ncbi_disease_ner_tuned_on_TAC2017")


X_train = tokenizer(
    text=X_train['Impression'].tolist(),
    add_special_tokens=True,
    max_length=158,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)

X_test = tokenizer(
    text=X_test['Impression'].tolist(),
    add_special_tokens=True,
    max_length=158,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)

X_val = tokenizer(
    text=X_val['Impression'].tolist(),
    add_special_tokens=True,
    max_length=158,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)


for parameter in dict_parameters:
    print(parameter)
    print('Training')
    with mlflow.start_run(): 
        mlflow_log_parameters(parameter) 
        bert_model = create_bert(bert, parameter['learning_rate'], parameter['dropout'])
        hist = bert_model.fit( x = {'input_ids':X_train['input_ids'], 'attention_mask':X_train['attention_mask']}, 
                     y=y_train, epochs=parameter['epochs'], batch_size=parameter['batch_size'])
    
        training_test_mae( mlflow, hist)
        
        print('Testing')
        split_type = 'test'
        y_pred = bert_model.predict(x = {'input_ids':X_test['input_ids'], 'attention_mask':X_test['attention_mask']})
        accuracy, precision, recall, f1score = evaluate(y_test, y_pred, mlflow, split_type)
        
        print('Validation')
        split_type = 'validation'
        y_pred = bert_model.predict(x = {'input_ids':X_val['input_ids'], 'attention_mask':X_val['attention_mask']})
        accuracy, precision, recall, f1score = evaluate(y_val, y_pred, mlflow,  split_type)
        
        mlflow.end_run()