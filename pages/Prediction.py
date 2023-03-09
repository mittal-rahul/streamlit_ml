import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np

import time

import pandas as pd
import logging
import spacy
import numpy as np
import datetime

from common_function_library import *
from main_model_inference import *
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.markdown("# Prediction")
st.markdown("#### Predicting categories of user's queries")
st.sidebar.markdown("# Prediction ")
with st.spinner("Loading..."):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info('Start of inferencing page')


    storage_account_name = None
    container_name = None
    sas_key = None
    mount_point = None
    conn_string = None

    df_models = pd.read_csv('pages/model_logs.csv')

    model_names = dict(
        df_models.sort_values(by='train_model_run_date',
                            ascending=False,
                            ignore_index=True).iloc[0])
    logger.info(model_names)

     
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


    # folder_path is path where Enquiry Ranking Mappings are present in csv form
    folder_path = "config_data/"
    # directory is path where saved models are present in pickle format
    directory = "models/"

    #threshold = 0.75  # Threshold is the minimum probability under which predictions will be marked as "Uncertain" by lr model
    threshold_or = 0.80  # Threshold is the minimum probability under which predictions will be marked as "Uncertain" by OR model


    # Main model and vectorized vocab name to make predictions

    lr_main_model_name = model_names['main_model_name']
    main_vectorized_vocab_name = model_names['main_vectorized_vocab_name']

    # Whole path along with model name to load main models
    lr_model_load_path = directory + lr_main_model_name
    main_vectorized_vocab_load_path = directory + main_vectorized_vocab_name

    #Mapping files for ranking
    score_file = "score_mapping.csv"
    param_imp_file = "parameter_importance.csv"
    profile_map_file = "profile_mapping.csv"
    keywords_category_file = "keywords_for_category.csv"


    # Loading Spacy en_core_web_sm package
    nlp = spacy.load('en_core_web_sm')
    # Loading Spacy stopwords
    sw_spacy = nlp.Defaults.stop_words
    # Deleting selected stopwords from spacy list so that they are not removed from input text.
    sw_spacy -= {
        "without",
        "why",
        "what",
        "when",
        "whose",
        "whom",
        "serious",
        "over",
        "other",
        "n‘t",
        "n’t",
        "n't",
        "not",
        "nor",
        "neither",
        "no",
        "nobody",
        "none",
        "noone",
        "never",
        "less",
        "least",
        "empty",
        "cannot",
        "another",
        "amount",
        "against",
        "get",
        "about",
    }
    # Adding stopwords to spacy so that they can be removed from input text
    sw_spacy.update({
        "test",
        "good",
        "morning",
        "afternoon",
        "evening",
        "night",
        "sir",
        "sirs",
        "dear",
        "hello",
        "gmail",
        "com",
    })





try:
    uploaded_file = st.file_uploader("Upload file for predictions", type="csv")
    original_df = pd.DataFrame()
    if uploaded_file is not None:
        original_df = pd.read_csv(uploaded_file)
    #st.write(original_df[0:10].to_html(index=False),unsafe_allow_html=True)
    gb = GridOptionsBuilder.from_dataframe(original_df)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    #gb.configure_side_bar() #Add a sidebar
    #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()


    AgGrid(
        original_df,
        gridOptions=gridOptions,
        # data_return_mode='AS_INPUT', 
        update_mode='NO_UPDATE', 
        fit_columns_on_grid_load=True,
        #enable_enterprise_modules=True,
        height=350, 
        width='100%',
        #reload_data=True
    )
except Exception as e:
    logger.error(
        'Reading data failed',
        exc_info=e)
    raise e

st.write("")
st.write("")
try:
    threshold = 0.75  # Threshold is the minimum probability under which predictions will be marked as "Uncertain" by lr model
    threshold = st.slider('Select the minimum probability under which predictions will be marked as "Uncertain" by model',min_value=0.0, max_value=1.0, value=.75)
    if st.button('Predict'):

        
        with st.spinner('In Progress.. Hold on, It may take a while...'):


            
            original_df.fillna("", inplace=True)

            # Assigning unique_id to each row to handle data merging and feeding to multiple models
            id = np.arange(len(original_df))
            original_df.insert(0, "unique_id", id)

            # Selecting dataset which has some text in Message colum for inferencing from main LR model
            df_desc = original_df[original_df["Description"].str.strip() != ""]

            # object for MainModelInference
            main_model_obj = MainModelInference()

            # Making predictions for Purchasing Information, Category Information and Other
            # df is input pandas dataframe for inferencing, threshold is float(0.0-1.0), lr_model_load_path &  main_vectorized_vocab_load_path are paths to load model and serialized vocabulary
            try:
                output_df = main_model_obj.main_model_predict(
                    nlp,
                    sw_spacy,
                    df_desc,
                    lr_model_load_path,
                    main_vectorized_vocab_load_path,
                    threshold,
                    conn_string,
                    container_name,
                    in_blob_storage="",
                )
            except Exception as e:
                logger.error("Error in predicting from main model")
                raise e

            # output_df has all the df columns and added columns as "clean_translated_Message", "unigrams", "bigrams", "trigrams", "ngrams","pred_label", "probability","calculated_label"

            output_df.fillna("", inplace=True)
            # logger.info("Inferenced categories count: " +
            #         str(output_df["calculated_label"].value_counts()))

            # Adding data to inference blob container
            df_combined = output_df.fillna("")
            df_combined = df_combined.drop(
                columns=['unigrams', 'bigrams', 'trigrams', 'ngrams'])
        
        st.write('')
        st.write('')
        st.write('Predictions of queries')

        df_combined = df_combined[['Description', 'calculated_label', 'probability']]
        df_combined['probability']=df_combined['probability'].astype(float).round(decimals=4)
        df_combined= df_combined.rename(columns={'calculated_label':'Predicted Category','Description':'Description','probability':'Probability'})
        gb = GridOptionsBuilder.from_dataframe(df_combined)
        gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
        gb.configure_side_bar() #Add a sidebar
        #gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
        gridOptions = gb.build()

        AgGrid(
            df_combined,
            gridOptions=gridOptions,
            # data_return_mode='AS_INPUT', 
            update_mode='NO_UPDATE', 
            fit_columns_on_grid_load=True,
            #enable_enterprise_modules=True,
            height=350, 
            width='100%',
            #reload_data=True
        )
        #st.write(df_combined[0:10].to_html(index=False),unsafe_allow_html=True)

        st.balloons()




        csv = convert_df(df_combined)
        st.write("")
        df_size= df_combined.shape[0]
        uncertain_count = df_combined[df_combined['Predicted Category']=="Uncertain"].shape[0]
        confidence_score = ((df_size-uncertain_count)/df_size)*100
        st.write("Model accuracy for probability",threshold," is: ",confidence_score,"%" )
        



        st.download_button(
        "Download Predictions",
        csv,
        "Prediction.csv",
        "text/csv",
        key='download-csv'
        )

        #st.write(df_combined[0:10].to_html(index=False),unsafe_allow_html=True)
        # with st.spinner("Saving Predicted data as prediction.csv"):

        #     try:
        #         df_combined.to_csv("predictions.csv")
        #         logger.info("Enquiry category predictions written to predictions.csv")
        #     except Exception as e:
        #         logger.error("Enquiry category predictions cannot be written", exc_info=e)
        #         raise e


        st.success('Prediction completed!!!!')
        #st.write('now you may proceed to model training or model eda..')
except Exception as e:
    st.write("Cannot make predictions. This may be due to one of the reasons:")
    st.write("1. Any trained model is not present.")
    st.write("2. The input data for predictions is not valid or blank.")
    logger.error("Error in making predictions", exc_info=e)
