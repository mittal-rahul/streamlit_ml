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
st.set_page_config(layout="wide")
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time
import spacy
import datetime
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.markdown("# Model Training")
st.markdown("#### Training the machine learning model on preprocessed user queries and categories.")
st.sidebar.markdown("# Model Training ")



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
        #initialize & fit model
        model = LogisticRegression(
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
        return model.fit(X, Y)

        # model = keras.models.Sequential()
        # model.add(keras.layers.InputLayer(input_shape=X.shape[1]))
        # model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(300, activation="relu"))
        # model.add(keras.layers.Dense(100, activation="relu"))
        # model.add(keras.layers.Dense(3, activation="softmax"))

        # #st.write("shape",X)

        # # X = np.asarray(X)
        # # Y = np.asarray(Y).astype('float32')
        # # x_test = np.asarray(x_test)
        # # y_test = np.asarray(y_test).astype('float32')

        # model.compile(loss='categorical_crossentropy', optimizer= "adam", metrics=["accuracy"])
        # #model = RandomForestClassifier(max_features= 'log2',min_samples_leaf= 1,n_estimators= 500)
        # model = model.fit(X, tf.keras.utils.to_categorical(Y,3), epochs=50, validation_data=(x_valid,  tf.keras.utils.to_categorical(y_valid,3)))
        
        # return model

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
        



        
        def literal_return(val):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError) as e:
                return val

        df['ngrams'] = df.ngrams.apply(lambda x: literal_return(x)) 

        X = df.drop("Category", axis=1)
        Y = df["Category"]

        # le = preprocessing.LabelEncoder()

        # Y = le.fit_transform(Y)

        # Splitting the training data into train and test data:

        df_train, df_test, Y_train, Y_test = train_test_split(
            X, Y, stratify=Y, random_state=42, test_size=test_size)

        #df_train = df_train.toarray()

        #df_test = df_test.toarray()
        
        df_train["Category"] = Y_train
        df_test["Category"] = Y_test


        

        # Creating object of class TFIDFVectorizer to initialize tfidf vectorizer and vectorize the training dataset
        vect = TFIDFVectorizer()

        custom_vocab = vect.create_vocab(df_train["ngrams"])
        vectorizer = vect.corpus_vectorizer(custom_vocab)
        corpus_vectors = vectorizer.fit_transform(df_train["ngrams"])
        # getting the vectorized vocabulary from the vectorizer to serialize and store as pickle file
        trained_vocabulary = vectorizer.vocabulary_
        #st.write(trained_vocabulary)
        #st.write(df_train.medical_specialty.value_counts())

        #X_train, X_valid, y_tr, y_valid = train_test_split(corpus_vectors, Y_train, stratify=Y_train, test_size=0.1, random_state =42)

        # Using the ADASYN sampling strategy to resolve the biasness in dataset:
        ada = ADASYN(sampling_strategy="auto",
                     random_state=42,
                     n_neighbors=5,
                     n_jobs=-1)
        X_resampled, y_resampled = ada.fit_resample(corpus_vectors, Y_train)

        # X_train = X_train.toarray()
        # X_valid = X_valid.toarray()


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

if True:

    # '''if 'data_loaded' not in st.session_state:
    #     st.write("Data is not loaded and Preprocessed in Main Page. Please load and preprocess data from Main Page first.")
    # else:'''
    with st.spinner("Loading..."):
    #from main_model_training import *
        from common_function_library import *
        #test_size = st.slider('Select the test size for evaluating trained model',min_value=0.1, max_value=0.3, value=0.2)
        test_size = 0.2
        if st.button('Train Model'):

            with st.spinner("In Progress..."):
                train_start_time = datetime.datetime.now()
                # using now() to get current time
                model_time = datetime.datetime.now()

                    
                directory = "models/"  # Path to dump the trained models
                data_path = "input_data/"  # Path to read the input training dataset

                # Main Model Parameters
                # Creating date_time_string with MM_DD_YYYY_HH_MM format
                date_time_string = (str(model_time.month) + "_" + str(model_time.day) + "_" +
                                    str(model_time.year) + "_" + str(model_time.hour) + "_" +
                                    str(model_time.minute))

                # Creating model and vectorized vocab name with prefix date_time_string
                lr_main_model_name = date_time_string + "_" + "lr_main_model.pkl"
                main_vectorized_vocab_name = date_time_string + "_" + "main_vectorized_vocab.pkl"

                lr_model_save_path = (directory + lr_main_model_name
                                    )  # path and name where main model has to be saved
                main_vectorized_vocab_save_path = (
                    directory + main_vectorized_vocab_name
                )  # path and name where main vectorized vocabulary has to be saved

                threshold = 0.75  # Threshold is the minimum probability under which predictions will be marked as "Uncertain"
                #test_size = 0.2  # Test size is the fraction of training data to be used fortesting &  metrics calculation by model


                storage_account_name = None
                container_name = None
                sas_key = None
                mount_point = None
                conn_string = None

                new_df = pd.read_csv('cleaned_csv.csv')


                main_model_obj = MainModelTraining()

                # df is input pandas preprocessed dataframe, threshold is float(0.0-1.0), lr_model_save_path &  main_vectorized_vocab_save_path are paths

                try:
                    output_df, metrics_dict, conf_matrix = main_model_obj.main_model(
                        new_df,
                        threshold,
                        lr_model_save_path,
                        main_vectorized_vocab_save_path,
                        test_size,
                        conn_string=conn_string,
                        container_name=container_name,
                        in_blob_storage = None,
                    )
                    logger.info("metrics dictionary: "+ str(metrics_dict))
                
                # output_df has all the df columns and added columns as "clean_translated_Message", "unigrams", "bigrams", "trigrams", "ngrams","pred_label", "probability","calculated_label"
                # metrics_dict is dictionary containing model's metrics "precision", "accuracy", "recall", "f1-score", "roc_auc"
                # conf_matrix has the confusion matrix in form of n-d array.


                
                #logger.info("confusion matrix: "+str(conf_matrix))

                    # Calculating total time taken by model training
                    train_end_time = datetime.datetime.now()
                    total_run_time = train_end_time - train_start_time  ##convert into minutes
                    # Other parameters to log model details
                    training_data_size = new_df.shape[0]

                    metrics_dict_df  = pd.DataFrame(metrics_dict,index=[0])
                    metrics_dict_df=metrics_dict_df.round(decimals=4)
                    st.write("")
                    st.write("")
                    st.write("Training Metrics")
                    AgGrid(
                        metrics_dict_df,
                        #gridOptions=gridOptions,
                        # data_return_mode='AS_INPUT', 
                        update_mode='NO_UPDATE', 
                        fit_columns_on_grid_load=True,
                        #enable_enterprise_modules=True,
                        height=100, 
                        #width='100%',
                        reload_data=False
                    )

                    #st.write(round(metrics_dict,4)[0:10].to_html(index=False),unsafe_allow_html=True)

                    st.write("")
                    st.write("")
                    st.write("Confusion Matrix")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                    
                    fig, ax = plt.subplots(figsize=(3,3))
                    
                    col1,col2 = st.columns(2)
                    with col1:
                        plt.rc('font', size=4)          # controls default text sizes
                        plt.rc('axes', titlesize=4)     # fontsize of the axes title
                        plt.rc('axes', labelsize=5)    # fontsize of the x and y labels
                        plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
                        plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
                        plt.rc('legend', fontsize=4)    # legend fontsize
                        plt.rc('figure', titlesize=5)
                        #plt.rcParams['figure.figsize'] = (2, 1)
                        #conf_matrix.plot()
                        conf_matrix.plot(ax=ax)
                        st.pyplot()#figsize=(4, 16))
                    #st.write(pd.DataFrame(conf_matrix)[0:10].to_html(index=False,header= False),unsafe_allow_html=True)

                    model_log_dict = {
                    "main_model_name": [lr_main_model_name],
                    "main_vectorized_vocab_name": [main_vectorized_vocab_name],
                    "train_model_run_date": [train_start_time],
                    "total_run_time": [total_run_time],
                    "training_data_size": [training_data_size],
                    "algorithm_used": "Logistic Regression"
                    }
                    # "main_model_metrics": [str(metrics_dict)],
                    # }

                    model_log_df = pd.DataFrame.from_dict(model_log_dict)
                    model_log_df.to_csv("pages/model_logs.csv", index=False)
                    logger.info("model_logs written in csv format")
                
                    st.write("")
                    st.write("")
                    st.write("Model Summary")
                    # gb = GridOptionsBuilder.from_dataframe(model_log_df)
                    # # gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                    # # gb.configure_side_bar() #Add a sidebar
                    # #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                    # gridOptions = gb.build()

                
                    AgGrid(
                        model_log_df,
                        #gridOptions=gridOptions,
                        # data_return_mode='AS_INPUT', 
                        update_mode='NO_UPDATE', 
                        fit_columns_on_grid_load=True,
                        #enable_enterprise_modules=True,
                        height=100, 
                        #width='100%',
                        reload_data=False
                    )
                    #st.write(model_log_df[0:10].to_html(index=False),unsafe_allow_html=True)
                    #st.write("")
                    st.write("")

                    st.success('Model trained successfully!')

                    st.balloons()

                except Exception as e:
                    logger.error("Error in training model", exc_info=e)
                    if "No samples will be generated with the provided ratio settings" in str(e):
                        st.error("Input data size is very less. Either select the test ratio less or load another file with more data on Main Page.")
