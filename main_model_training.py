#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import dill as pickle
import logging

import streamlit as st
import pandas as pd
import numpy as np

import time

import pandas as pd
import logging
import spacy
import numpy as np
import datetime


import streamlit as st


# In[4]:

#from main_model_training import *
from common_function_library import *


import seaborn as sns
import matplotlib.pyplot as plt



# In[ ]:


logger = logging.getLogger(__name__)


# In[5]:


class MainModelTraining:
    """
    Class to train the mail model on training data for labels "Purchasing Information", "Category Information" and "Other".
    """

    # Function to initialize selected Logistic Regression model for predictions of Purchasing Infromation,
    # Category Information and Other Enquiry category.

    def fit_main_model(self, X, Y):
        """
        -------
        input: vectorized training vectors and the target variable
        -------
        Train the model on input dataset
        return: trained model, else return None
        -------
        """
        # initialize & fit model
        lr = LogisticRegression(
            C=4714.8663634573895,
            dual=False,
            fit_intercept=True,
            intercept_scaling=2,
            l1_ratio=None,
            max_iter=200,
            multi_class="ovr",
            n_jobs=-1,
            penalty="l2",
            random_state=42,
            solver="saga",
            tol=0.0001,
            verbose=0,
            warm_start=False,
        )
        return lr.fit(X, Y)

    def main_model(
        self,
        df,
        threshold,
        lr_model_save_path,
        main_vectorized_vocab_save_path,
        test_size,
        conn_string=None,
        container_name=None,
        in_blob_storage=None,
    ):
        """
        This function is the main function of the model training. It applies the preprocessing on input dataset,
        then vectorizes and then trains the model on it.
        -------
        :parms nlp(spacy object): spacy object for preprocessing
        :parms sw_spacy(spacy stopwords): customized list of spacy stopwords
        :parms df(pandas DataFrame):input training dataset
        :parms threshold(float):threshold of probability to mark the category of predictions
                                otherwise will be marked as "Uncertain"
        :parms lr_model_save_path(str):Path to save the trained lr model
        :parms main_vectorized_vocab_save_path(str):Path to save the trained vectorizer vocabulary
        :param test_size (float): test_size for model training
        -------

        return pandas dataframe of training dataset with added columns
        ["unigrams", "bigrams", "trigrams", "ngrams", "pred_label", "probability", "calculated_label"]
        along with metrics_dict which is dictionary of metrics and conf_matrix (confusion matrix)
        -------
        """
        
        X = df.drop("Category", axis=1)
        Y = df["Category"]

        # Splitting the training data into train and test data:

        df_train, df_test, Y_train, Y_test = train_test_split(
            X, Y, stratify=Y, random_state=42, test_size=test_size)
        df_train["Category"] = Y_train
        df_test["Category"] = Y_test

        # Creating object of class TFIDFVectorizer to initialize tfidf vectorizer and vectorize the training dataset
        vect = TFIDFVectorizer()

        custom_vocab = vect.create_vocab(df_train["ngrams"])
        vectorizer = vect.corpus_vectorizer(custom_vocab)
        corpus_vectors = vectorizer.fit_transform(df_train["ngrams"])
        # getting the vectorized vocabulary from the vectorizer to serialize and store as pickle file
        trained_vocabulary = vectorizer.vocabulary_

        # Using the ADASYN sampling strategy to resolve the biasness in dataset:
        ada = ADASYN(sampling_strategy="auto",
                     random_state=42,
                     n_neighbors=5,
                     n_jobs=-1)
        X_resampled, y_resampled = ada.fit_resample(corpus_vectors, Y_train)

        # Fitting the Logistic Regression model using the fit_main_model function:
        try:
            lr = self.fit_main_model(X_resampled, y_resampled)
        except Exception as e:
            logger.error('Model Fit Error: ', exc_info=e)
            raise e

        # Creating object of class CalculateMetrics to evaluate the trained model:
        calculate_metrics = CalculateMetrics()

        # Getting test dataset with added colums pred_label, probability,calculated_label.
        # It also returns dictionary of metrics and confusion_matrix
        try:
            output_df, metrics_dict, conf_matrix = calculate_metrics.metrics_calculation(
                threshold, df_test, vectorizer, lr, "ovr")
        except Exception as e:
            logger.error('Error in  metrics calculation of main model: ',
                         exc_info=e)
            raise e

        # Initializing class DataLoadDump to call the dump_data function and dump the trained vectorizer and trained model:
        model_dump = DataLoadDump()

        # Dumping trained logistic regression model as pickle file at path name as lr_model_save_path
        try:
            if model_dump.dump_data(lr_model_save_path, lr, conn_string,
                                    container_name, in_blob_storage):
                logger.info('Main LR model dumped successfully')
            else:
                logger.error("Main LR model cannot be dumped")
                raise Exception("Main LR model cannot be dumped")
        except Exception as e:
            logger.error('Model dumping Error:', exc_info=e)
            raise e

        # Dumping vectorizer as pickle file at path name as main_vectorized_vocab_save_path
        try:
            if model_dump.dump_data(
                    main_vectorized_vocab_save_path,
                    trained_vocabulary,
                    conn_string,
                    container_name,
                    in_blob_storage,
            ):
                logger.info('Main vectorized vocabulary dumped successfully')
            else:
                logger.error("Main vectorized vocabulary cannot be dumped")
                raise Exception("Main vectorized vocabulary cannot be dumped")
        except Exception as e:
            logger.error('Main Vocab dumping Error: ', exc_info=e)
            raise e

        return output_df, metrics_dict, conf_matrix


# In[ ]:



