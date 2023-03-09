from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import streamlit as st
st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from ast import literal_eval

import warnings
warnings.filterwarnings("ignore")


st.markdown("# Exploratory Data Analysis")
st.markdown("#### Provides exploratory data analysis for category distribution and frequency of words in queries ")
st.sidebar.markdown("# EDA ")
if True:
    # '''if 'data_loaded' not in st.session_state:
    #     st.write("Data is not loaded and Preprocessed in Main Page. Please load and preprocess data from Main Page first.")
        
    # else:'''
    with st.spinner("Loading..."):
        df = pd.read_csv('cleaned_csv.csv')

        def literal_return(val):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError) as e:
                return val

        df['unigrams'] = df.unigrams.apply(lambda x: literal_return(x)) 
        df['bigrams'] = df.bigrams.apply(lambda x: literal_return(x)) 
        df['trigrams'] = df.trigrams.apply(lambda x: literal_return(x)) 

        class EDA:
            def get_high_frequency_words(self, text):
                """
                Returns a dataframe with unique words and the count of its occurances
                """
                exclude_words = ["like", "thank"]  # words to exclude from o/p
                if type(text) == list:
                    try:
                        text = " ".join(text)
                    except Exception as e:
                        text = [subitem for item in text for subitem in item]
                        text = " ".join(text)

                text = text.lower()
                words_dict = {}
                words_list = text.split()
                for num in range(0, len(words_list)):
                    if words_list[num] not in words_dict.keys():
                        if (words_list[num] not in exclude_words) & (
                                words_list[num].isnumeric() == False):
                            words_dict[words_list[num]] = 1
                    else:
                        if (words_list[num] not in exclude_words) & (
                                words_list[num].isnumeric() == False):
                            words_dict[words_list[num]] += 1
                freq_df = pd.DataFrame({
                    "word": words_dict.keys(),
                    "freq": words_dict.values()
                }).sort_values(by="freq", ascending=False)
                return freq_df

            def gen_word_cloud(self, words: list):
                """
                This function generates a word cloud for the list of words given as input
                words [list] : This is a list of words with which wordlcoud has to be created

                """
                wordcloud_words = []
                # checking if list is being read as string
                if len(words) == len([item for item in words if type(item) == str]):
                    wordcloud_words.append([(subword) for word in words
                                            for subword in eval(word)])
                else:
                    wordcloud_words.append([(subword) for word in words
                                            for subword in (word)])

                #         wordcloud_words = " ".join(list(set(wordcloud_words[0])))
                wordcloud_words = " ".join(wordcloud_words[0])
                wordcloud = WordCloud(
                    width=1250,
                    height=1000,
                    #max_words = word_count,
                    background_color="white",
                    # stopwords = stopwords,
                    #min_font_size=10,
                    #relative_scaling  = scale
                ).generate(wordcloud_words)

                # plot the WordCloud image
                plt.figure(facecolor=None)
                try:
                    plt.imshow(wordcloud)
                except Exception as e:
                    plt.imshow(wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.show()
                st.pyplot()#figsize=(8, 18))

            def distribution_graph(self, category_dict: dict, title: str):
                """
                This function generates a bar graph
                ----
                :param category_dict (dict): dictionary of unique categroies and their values
                :param title (str): title of the generated bar graph

                """

                plt.figure(figsize=(8, 18), facecolor=None)

                fig = px.bar(
                    x=category_dict.keys(),
                    y=category_dict.values(),
                    title=title,
                    text_auto=True,
                    labels={
                        "x": "Categories",
                        "y": "Count"
                    },
                )
                fig.update_traces(textfont_size=12,
                                textangle=0,
                                textposition="outside")
                #fig.show()
                st.plotly_chart(fig)

            def word_length_count(self, column):
                """
                This function generates a bar graph between the length of the word and count of words for each length
                """

                words = []
                words.append([subword for word in column for subword in word])
                words = words[0]
                words = set(words)
                #print(words)

                len_dict = {}
                for word in words:
                    length = len(word)

                    if length not in len_dict.keys():
                        if word.isnumeric() == True:
                            pass
                        else:
                            len_dict[length] = [word]
                    else:
                        if word.isnumeric() == True:
                            pass
                        else:
                            len_dict[length].append(word)

                word_len_df = pd.DataFrame({
                    "Char Length": len_dict.keys(),
                    "Words": len_dict.values()
                })
                word_len_df["No of words"] = word_len_df["Words"].apply(len)

                plt.figure(figsize=(8, 18), facecolor=None)
                fig = px.bar(
                    word_len_df,
                    x="Char Length",
                    y="No of words",
                    title="Character Length vs Count of Words",
                    hover_data=["Char Length", "No of words"],
                    color="No of words",
                    height=450,
                    text_auto=True,
                )
                fig.update_traces(textfont_size=12,
                                textangle=0,
                                textposition="outside")
                #fig.show()
                st.plotly_chart(fig)

        



        eda_obj = EDA()

    with st.spinner("Loading..."):
        category_dict = df['Category'].value_counts().to_dict()
        eda_obj.distribution_graph(category_dict, "Category Distribution")

        # st.write("High Frequency Words - Entire Corpus")
        # st.write(eda_obj.get_high_frequency_words(df[df['clean_translated_Message']!=""]['clean_translated_Message'].astype(str).tolist()))

        st.write("Most occuring Words - All categories")

        tab1, tab2 = st.tabs(["Unigrams", "Bigrams"])

    with st.spinner("Loading..."):
        with tab1:
            with st.container():

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Most occuring Unigrams")
                    AgGrid(
                    eda_obj.get_high_frequency_words(df[df['unigrams']!=""]['unigrams'].tolist()),
                    #gridOptions=gridOptions,
                    # data_return_mode='AS_INPUT', 
                    update_mode='NO_UPDATE', 
                    fit_columns_on_grid_load=True,
                    #enable_enterprise_modules=True,
                    height=350, 
                    #width='100%',
                    reload_data=False
                    )
                    #st.write(eda_obj.get_high_frequency_words(df[df['unigrams']!=""]['unigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
                    st.write("")
                    st.write("")
                with col2:
                    #st.write("Unigrams Wordcloud")
                    eda_obj.gen_word_cloud(df['unigrams'].tolist())


        with tab2:
            with st.container():

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Most occuring Bigrams")
                    AgGrid(
                    eda_obj.get_high_frequency_words(df[df['bigrams']!=""]['bigrams'].tolist()),
                    #gridOptions=gridOptions,
                    # data_return_mode='AS_INPUT', 
                    update_mode='NO_UPDATE', 
                    fit_columns_on_grid_load=True,
                    #enable_enterprise_modules=True,
                    height=350, 
                    #width='100%',
                    reload_data=False
                    )
                    #st.write(eda_obj.get_high_frequency_words(df[df['bigrams']!=""]['bigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
                    st.write("")
                    st.write("")
                with col2:
                    #st.write("Bigrams Wordcloud")
                    eda_obj.gen_word_cloud(df['bigrams'].tolist())

        # with tab3:
        #     with st.container():
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             st.write("High Frequency Words - Trigrams")
        #             st.write(eda_obj.get_high_frequency_words(df[df['trigrams']!=""]['trigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
        #             st.write("")
        #             st.write("")
        #         with col2:
        #             #st.write("Trigrams Wordcloud")
        #             eda_obj.gen_word_cloud(df['trigrams'].tolist(),word_count = 30,scale =0)

        st.write("Category wise words")

        tab1, tab2 = st.tabs(["Unigrams", "Bigrams"])

    with st.spinner("Loading..."):
        for i in df.Category.unique():
            with tab1:
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Most occuring Unigrams in category: ",i)
                        AgGrid(
                        eda_obj.get_high_frequency_words(df[df['Category']==i]['unigrams'].tolist()),
                        #gridOptions=gridOptions,
                        # data_return_mode='AS_INPUT', 
                        update_mode='NO_UPDATE', 
                        fit_columns_on_grid_load=True,
                        #enable_enterprise_modules=True,
                        height=350, 
                        #width='100%',
                        reload_data=False
                        )
                        #st.write(eda_obj.get_high_frequency_words(df[df['medical_specialty']==i]['unigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
                        st.write("")
                        st.write("")
                    with col2:
                        #st.write("Unigrams Wordcloud")
                        eda_obj.gen_word_cloud(df[df['Category']==i]['unigrams'].tolist())
                    

            with tab2:
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Most occuring Bigrams in category: ",i)
                        AgGrid(
                        eda_obj.get_high_frequency_words(df[df['Category']==i]['bigrams'].tolist()),
                        #gridOptions=gridOptions,
                        # data_return_mode='AS_INPUT', 
                        update_mode='NO_UPDATE', 
                        fit_columns_on_grid_load=True,
                        #enable_enterprise_modules=True,
                        height=350, 
                        #width='100%',
                        reload_data=False
                        )
                        #st.write(eda_obj.get_high_frequency_words(df[df['medical_specialty']==i]['bigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
                        st.write("")
                        st.write("")
                    with col2:
                        #st.write("Bigrams Wordcloud")
                        eda_obj.gen_word_cloud(df[df['Category']==i]['bigrams'].tolist())

                        
                # with tab3:
                #     with st.container():
                #         col1, col2 = st.columns(2)
                #         with col1:
                #             st.write("High Frequency Trigrams in category: ",i)
                #             st.write(eda_obj.get_high_frequency_words(df[df['medical_specialty']==i]['trigrams'].tolist())[0:10].to_html(index=False),unsafe_allow_html=True)
                #             st.write("")
                #             st.write("")

                #         with col2:
                #             st.write("Trigrams Wordcloud")
                #             eda_obj.gen_word_cloud(df[df['medical_specialty']==i]['trigrams'].tolist(),word_count = 30,scale =0)
                        

    with st.spinner("Loading..."):


        def literal_return(val):
            try:
                return literal_eval(val)
            except (ValueError, SyntaxError) as e:
                return val

        #df['unigrams_eval'] = df.unigrams.apply(lambda x: literal_return(x)) 


    with st.spinner("Loading..."):

        eda_obj.word_length_count(df['unigrams'])