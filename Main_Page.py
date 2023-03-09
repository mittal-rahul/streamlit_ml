import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

import pandas as pd
import logging
import spacy
import numpy as np
import datetime
#import sweetviz as sv
import streamlit.components.v1 as components
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

st.markdown("# Categorization")
st.markdown("#### This Machine learning engine is capable of data analysis and trains model on text queries. The trained model can predict category of user queries.")
st.sidebar.markdown("# Main page ")
with st.spinner("Please wait!!!"):

    #from main_model_training import *
    from common_function_library import *

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info('Start of training pipeline')


    storage_account_name = None
    container_name = None
    sas_key = None
    mount_point = None
    conn_string = None

    # Loading Spacy en_core_web_sm package
    nlp = spacy.load('en_core_web_sm')
    # Loading Spacy stopwords
    sw_spacy = nlp.Defaults.stop_words
    # Deleting selected stopwords from spacy list so that they are not removed from input text.
    #spacy_df = pd.read_csv('pages/high frequency_common_unigrams.csv')
    #sw_spacy.update(spacy_df ['word'].to_list())
if 'data_loaded' not in st.session_state:
    with st.spinner("Please wait!!!"):
        try:
            uploaded_file = st.file_uploader("Upload data for model training", type="csv")
            original_df = pd.DataFrame()

            #report = sv.analyze(original_df)
            
            #report.show_html('analyze.html',open_browser = False)

            #st.write(obj1)
            #components.html(str(obj1), width=1100, height=1200, scrolling=True)
            #with open('analyze.html') as html:
                #st.write(html.read())
                #components.html(str(html.read()), width=1100, height=1200, scrolling=True)

            if uploaded_file is not None:
                original_df = pd.read_csv(uploaded_file)
                #original_df.reset_index(drop=True, inplace=True)
                

                st.write("Uploaded Data")

                #original_df = original_df.drop(columns=['keywords','transcription','sample_name'])
                #st.write(original_df[0:10].to_html(index=False),unsafe_allow_html=True)
                
                #st.dataframe(original_df)
                gb = GridOptionsBuilder.from_dataframe(original_df)
                gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                gb.configure_side_bar() #Add a sidebar
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
                st.write("")
                st.write("")
                    
                st.markdown("# Data Preprocessing")
                st.markdown("#### It cleanses the user's queries data and performs following actions: Removes non ASCII characters, numbers, stop words and punctuations from the input queries and generates unigrams, bigrams and trigrams.")

                if st.button('Preprocess Data'):

                    
                    with st.spinner('Hold on, it may take a while...'):
                
                    
                        original_df.fillna("", inplace=True)
                        # Filtering records having 'Message'
                        original_df = original_df[original_df["Description"].str.strip() != ""]

                        df = original_df.drop_duplicates(subset=["Description", "Category"])
                        # Removing the duplicate rows based on the translated_Message and Category columns


                        data_transform = DataTransformation(nlp, sw_spacy)


                        try:
                            new_df = data_transform.transform_data(df)
                        except Exception as e:
                            logger.error("Error in transforming data", exc_info=e)

                    


                    with st.spinner("Saving Preprocessed data as cleaned_df.csv"):

                        new_df.to_csv('cleaned_csv.csv')

                        
                        st.session_state['data_loaded'] = original_df
                        st.success('Data preprocessing completed!!!!')
                        st.write('You may now proceed to Model Training or Data Analysis page')
                        st.balloons()
            
        except Exception as e:
            logger.error(
                'Reading data failed',
                exc_info=e)
            raise e
else:
    with st.spinner("Please wait!!!"):    
        try:
            uploaded_file = st.file_uploader("Upload data for model training", type="csv")
            original_df = st.session_state.data_loaded
            if uploaded_file is not None:
                original_df = pd.read_csv(uploaded_file)
                del st.session_state['data_loaded']
                #original_df.reset_index(drop=True, inplace=True)

            st.write("Uploaded Data")

            #original_df = original_df.drop(columns=['keywords','transcription','sample_name'])
            #st.write(original_df[0:10].to_html(index=False),unsafe_allow_html=True)
            
            #st.dataframe(original_df)
            gb = GridOptionsBuilder.from_dataframe(original_df)
            gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
            gb.configure_side_bar() #Add a sidebar
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
            st.write("")
            st.write("")
                    
            st.markdown("# Data Preprocessing")
            st.markdown("#### It cleanses the user's queries data and performs following actions: Removes non ASCII characters, numbers, stop words and punctuations from the input queries and generates unigrams, bigrams and trigrams.")
            if uploaded_file is None:
                st.markdown("##### Above displayed data is already preprocessed. You may go to Train Model or Data Analysis page directly. You can preprocess data again.")
            if st.button('Preprocess Data'):

                
                with st.spinner('Hold on, it may take a while...'):
            
                
                    original_df.fillna("", inplace=True)
                    # Filtering records having 'Message'
                    original_df = original_df[original_df["Description"].str.strip() != ""]

                    df = original_df.drop_duplicates(subset=["Description", "Category"])
                    # Removing the duplicate rows based on the translated_Message and Category columns


                    data_transform = DataTransformation(nlp, sw_spacy)


                    try:
                        new_df = data_transform.transform_data(df)
                    except Exception as e:
                        raise
                        logger.error("Error in transforming data", exc_info=e)

                


                with st.spinner("Saving Preprocessed data as cleaned_df.csv"):

                    new_df.to_csv('cleaned_csv.csv')

                    
                    st.session_state['data_loaded'] = original_df
                    st.success('Data preprocessing completed!!!!')
                    st.write('You may now proceed to Model Training or Data Analysis page')
                    st.balloons()
                
        except Exception as e:
            logger.error(
                'Reading data failed',
                exc_info=e)
            raise e    
