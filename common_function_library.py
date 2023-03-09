#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import logging
import re
import string
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from spellchecker import SpellChecker
import dill as pickle
#from azure.storage.blob import BlobClient
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)


# In[ ]:





# In[ ]:

logger = logging.getLogger(__name__)

class PreProcessing:
    """
    This class cleans & returns the input text by removing stop words, punctuations,
    non-ascii characters etc.
    """
    def __init__(self, nlp, sw_spacy):
        self.nlp = nlp
        self.sw_spacy = sw_spacy
        self.spell = SpellChecker()

    def __remove_punctuations(self, text):
        """
        Returns text(i/o) from which punctuations have been replaced
        """
        trans = str.maketrans(string.punctuation, " " *
                              len(string.punctuation))  # replace with space
        return text.translate(trans)

    def __remove_sentences_length(self, text):
        """
        Returns text(i/o) if no of words in text >2 else returns ""
        """
        if len(text.split()) <= 2:
            return ""
        else:
            return text

    def __remove_stop_words(self, text):
        """
        Returns text(i/o) after removing stop words. Stop words are pre-defined.
        """
        return " ".join([
            word.lower() for word in text.split()
            if word.lower() not in self.sw_spacy
        ])

    def __remove_non_ascii(self, text):
        """
        Returns text(i/o) after removing non-ascii characters
        and extra spaces, tabs & new line characters.
        """
        text = str(text)
        text = re.sub(r"([^\x1F-\x7F]+)", "", text)
        text = text.replace("\n", " ").replace("\t", " ").replace(".", ". ")
        text = text.replace("_x000D_", "")
        text = re.sub("\\s+", " ", text)  # remove extra spaces
        if len(text) == 0:
            return ""
        else:
            return text

    def __fuzzy_update_mail_to_email(self, text):
        """
        Returns text(i/o) by replacing incorrect words with correct words
        """
        return text.replace("mail", "email").replace("eemail", "email")

    def __remove_meaningless_words(self, text):
        """
        Returns correct spelling of single word sentences
        """
        if len(text.split()) == 1:
            output = self.spell.correction(text)
            if (output == None) or (output == "i"):
                return ""
            else:
                return text
        else:
            return text
            # TODO - Two else conditions result in return of unaltered text. Condition only needs to return text once without else blocks.

    def __lematize(self, text):
        """
        Returns the lematized output of the input text
        """

        doc = self.nlp(text)  # converting to spacy doc
        return " ".join([token.lemma_ for token in doc])

    def __remove_only_numeric(self, text):
        """
        Returns "" if the entire text is numeric
        """
        # temp = "".join(text.split(" "))
        # if temp.isnumeric() == True:
        #     return ""
        # else:

        return  ''.join([i for i in text if not i.isdigit()])

    def __remove_words_of_length_less_than_3(self, text):
        """
        Returns text (i/o) after removing all charcters of length 1 & 2.
        """
        text = re.sub(r"\b[a-zA-Z]{1,2}\b", "", text)
        text = re.sub("\\s+", " ", text)
        return text
    
    def __remove_words_of_length_greater_than_20(self, text):
        """
        Returns text (i/o) after removing all words of length more than 20
        """
        text1 = [i for i in text.split() if len(i)<20]
        text1 = " ".join(text1)
        return text1

    def __remove_extra_x(self,text):

        redundant_words =  "xx"
        text = [word.lower() for word in text.split()] 
        text = [i for i in text if redundant_words not in i]
        return " ".join(text)
    
    def __clean_special_chars(self,text):

        result = re.sub(r'(?<=\w)-|-(?=\w)', '', text)
        result = re.sub(r'^https?:\/\/.*[\r\n]*', '', result)
        result = re.sub('<.*?>','',result)

        return result


    def clean_data(self, text):
        """
        This function cleans the data by applying various pre-determined cleaning steps
        Cleaning includes, removal of non-ascii  characters, punctuations, stop words etc.

        :param text (str): input text which has to be cleaned
        :return text (str): cleaned text

        """

        text = self.__remove_extra_x(text)
        text = self.__remove_non_ascii(text)
        text = self.__remove_punctuations(text)
        text = self.__remove_stop_words(text)
        text = self.__fuzzy_update_mail_to_email(text)
        text = self.__remove_meaningless_words(text)
        text = self.__lematize(text)
        text = self.__remove_only_numeric(text)
        text = self.__remove_words_of_length_less_than_3(text)
        text = self.__remove_words_of_length_greater_than_20(text)
        text = self.__clean_special_chars(text)
        return text


# In[ ]:


class GenerateNgrams:
    """
    This Class helps in creating ngrams by taking corpus
    from the user
    """
    def __init__(self, nlp):
        self.nlp = nlp

    def generate_ngrams(self, text, n):
        """
        This function applies fuzzy logic to return
        the top most match and its score

        :param text (str): input text from which ngrams have to be generated
        :param n (int): "n" in ngrams. [1 -> unigrams, 2 -> bigrams, 3 -> trigrams etc]
        :return ngrams (list): list of generated ngrams.

        """
        not_useful_unigrams = ["like", "thank",'not']
        ngrams = set()
        words_list = text.split()
        if n > 1:
            for num in range(0, len(words_list)):
                ngram = "_".join(words_list[num:num + n])
                if len(ngram.split("_")) == n:
                    ngrams.add(ngram)
        else:
            for num in range(0, len(words_list)):
                ngram = words_list[num]
                if ngram not in not_useful_unigrams:
                    ngrams.add(ngram)
        return list(ngrams)

    def rightBigrams(self, ngrams, pos_dict):
        """
        This function returns bigrams of combinations of
        noun types, adjective types and verb types

        :param ngrams (list): list of bigrams joined by "_"
        :param pos_dict (dict): dictionary of part of speech of each word in the ngrams
        :return bigrams (list): list of bigrams of defined combination

        """
        noun_types = ["NN", "NNS", "NNP", "NNPS"]
        adj_types = ["JJ", "JJR", "JJS"]
        verb_types = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        bigrams = []
        for bigram in ngrams:
            words = bigram.split("_")
            if pos_dict[words[0]] in adj_types and pos_dict[
                    words[1]] in noun_types:
                bigrams.append(bigram)
            elif pos_dict[words[0]] in noun_types and pos_dict[
                    words[1]] in noun_types:
                bigrams.append(bigram)
            elif pos_dict[words[0]] in noun_types and pos_dict[
                    words[1]] in verb_types:
                bigrams.append(bigram)
            elif pos_dict[words[0]] in verb_types and pos_dict[
                    words[1]] in noun_types:
                bigrams.append(bigram)
            else:
                pass
        return bigrams

    def rightTrigrams(self, ngrams, pos_dict):
        """
        This function returns bigrams of combinations of
        noun types, adjective types and verb types

        :param ngrams (list): list of trigrams joined by "_"
        :param pos_dict (dict): dictionary of part of speech of each word in the ngrams
        :return trigrams (list): list of trigrams of defined combination

        """

        noun_types = ["NN", "NNS", "NNP", "NNPS"]
        adj_types = ["JJ", "JJR", "JJS"]
        verb_types = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        trigrams = []
        for trigram in ngrams:
            words = trigram.split("_")
            if ((pos_dict[words[0]] in adj_types)
                    and (pos_dict[words[1]] in noun_types)
                    and (pos_dict[words[2]] in verb_types)):
                trigrams.append(trigram)
            elif ((pos_dict[words[0]] in noun_types)
                  and (pos_dict[words[1]] in verb_types)
                  and (pos_dict[words[2]] in noun_types)):
                trigrams.append(trigram)
            else:
                pass
        return trigrams

    def POS_Dictionary(self, text):
        """
        This function creates a dictionary of {<word>:<part of speech tag>}

        :param text (str): text for which PoS has to be found out
        :return pos_dict (dict): dictionary of {<word>:<part of speech tag>}

        """
        doc = self.nlp(text)
        pos_dict = {}
        for token in doc:
            pos_dict[token.text] = token.tag_
        return pos_dict


# In[ ]:


class DataTransformation:
    """
    This class returns dataframe after performing pre-requisite steps
    """
    def __init__(self, nlp, sw_spacy):
        self.preprocess = PreProcessing(nlp, sw_spacy)
        self.ngram_gen = GenerateNgrams(nlp)

    def transform_data(self, df):
        """
        Returns the dataframe after performing pre-requisite steps or exception in case of error
        """

        df.fillna("", inplace=True)
        df.replace(
            {
                "Not relevant": "Not Relevant",
                "Order related": "Order Related"
            },
            inplace=True,
        )
        with st.spinner("Cleaning data using Spell correction and removing stop words, non ascii characters, numbers & punctuations"):
            try:
                df["clean_translated_Message"] = df["Description"].apply(
                    lambda x: self.preprocess.clean_data(x))
            except Exception as e:
                logger.error('Text Cleaning Error: ', exc_info=e)
                raise e
            df.fillna("", inplace=True)
        # generating unigrams:
        with st.spinner("Generating unigrams"):
            try:
                df["unigrams"] = df["clean_translated_Message"].apply(
                    lambda x: self.ngram_gen.generate_ngrams(x, 1))
            except Exception as e:
                logger.error('Error in creating unigrams', exc_info=e)
                raise e
        # generating bigrams:
        with st.spinner("Generating bigrams"):
            try:
                df["bigrams"] = df["clean_translated_Message"].apply(
                    lambda x: self.ngram_gen.rightBigrams(
                        self.ngram_gen.generate_ngrams(x, 2),
                        self.ngram_gen.POS_Dictionary(x)))
            except Exception as e:
                logger.error('Error in creating bigrams', exc_info=e)
                raise e
        # generating trigrams
        with st.spinner("Generating trigrams"):
            try:
                df["trigrams"] = df["clean_translated_Message"].apply(
                    lambda x: self.ngram_gen.rightTrigrams(
                        self.ngram_gen.generate_ngrams(x, 3),
                        self.ngram_gen.POS_Dictionary(x)))
            except Exception as e:
                logger.error('Error in creating trigrams', exc_info=e)
                raise e
        # Joining Unigrams, Bigrams and Trigrams in a single colum to vectorize and further processing
        with st.spinner("Joining Unigrams, Bigrams and Trigrams"):
            try:
                df["ngrams"] = df.apply(
                    lambda x: x.unigrams + x.bigrams + x.trigrams,
                    axis=1,
                )
            except Exception as e:
                logger.error('Error in joining ngrams', exc_info=e)
                raise e
        df.fillna("", inplace=True)

        return df


# In[ ]:


class CustomLabelling:
    """
    This class returns custom label per class based on pre-defined threshold
    """
    def label_calculated(self, text, prob, threshold):
        """
        This function checks the probability of prediction and returns
        custom output based on the threshold

        :param text (str): predicted text/class
        :param prob (float): probability of the predicted text/class
        :param threshold (float): threshold of probability to mark the category of predictions
                                    otherwise will be marked as "Uncertain"
        :returns text if probability is above threshold else "Uncertain"

        """
        if prob >= threshold:
            return text
        else:
            return "Uncertain"


# In[ ]:


class TFIDFVectorizer:
    """
    This class returns the Tf-Idf vectors by generating the corpus
    from the "ngrams" column of the dataframe passed
    """
    def create_vocab(self, column):
        """
        Returns the list of custom vocab created from the "ngrams" column
        of the input dataframe
        """
        return list(set(chain.from_iterable(column)))

    def corpus_vectorizer(self, custom_vocab):
        """
        Returns the vectors of the column data that has been passed
        """
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            vocabulary=custom_vocab,
            analyzer="word",
            binary=False,
            decode_error="strict",
            dtype=np.float32,
            encoding="utf-8",
            input="content",
            lowercase=False,
            max_df=1.0,
            max_features=None,
            min_df=1,
            ngram_range=(1, 1),
            norm="l2",
            preprocessor=None,
            smooth_idf=True,
            stop_words=None,
            strip_accents=None,
            sublinear_tf=False,
            token_pattern=
            "(?u)\\b\\w\\w+\\b",  # selects all the words leaving out punctuations
            use_idf=False,
        )
        return vectorizer


# In[ ]:


class CalculateMetrics:
    """
    This class returns the dataframe with updated predictions, metrics and confusion matrix
    """
    def __vectorize_and_predict(self, df, text_vectorizer, model):
        """
        This function creates the vectors and predicts the output and returns the same
        """

        corpus_vectors = text_vectorizer.transform(df["ngrams"])
        preds = model.predict(corpus_vectors)
        df["pred_label"] = preds
        preds_proba = model.predict_proba(corpus_vectors)

        return preds, corpus_vectors, preds_proba, df

    def metrics_calculation(self,
                            threshold,
                            df,
                            text_vectorizer,
                            model,
                            multiclass=None):
        """ "
        This function is the evaluation function of the trained model. It applies the preprocessing on input dataset,
        then vectorizes it using trained vectorizer and then evaluates the model on it.
        -------
        :parms df(pandas DataFrame):input test dataset
        :parms model: trained model
        :parms vectorizer(TF-IDF Vectorizer): trained vectorizer
        :parms threshold(float):threshold of probability to mark the category of predictions otherwise will be marked as "Uncertain"
        :params multiclass (optional): multiclass category to be passed when predicting multiple classes
        -------

        return dataframe, metrics (dict), confusion matrix
        """

        preds, corpus_vectors, preds_proba, df = self.__vectorize_and_predict(
            df, text_vectorizer, model)

        df["probability"] = np.amax(preds_proba, axis=1)
        custom_label = CustomLabelling()
        df["calculated_label"] = df.apply(
            lambda x: custom_label.label_calculated(x["pred_label"], x[
                "probability"], threshold),
            axis=1,
        )

        metrics_dict = {}
        metrics_dict["accuracy"] = accuracy_score(df["Category"], preds)

        metrics_dict["precision"] = precision_score(df["Category"],
                                                    preds,
                                                    average="weighted")

        metrics_dict["recall_score"] = recall_score(df["Category"],
                                                    preds,
                                                    average="weighted")

        if multiclass == None:
            # Roc_Auc for OR model as it is binary classifier
            metrics_dict["roc_auc_score"] = roc_auc_score(
                df["Category"], preds_proba[:, 1])
        else:
            # Roc_Auc for main model as it is multi-class classifier
            metrics_dict["roc_auc_score"] = roc_auc_score(
                df["Category"], preds_proba, multi_class=multiclass)

        metrics_dict["f1_score"] = f1_score(df["Category"],
                                            preds,
                                            average="weighted")
        confusion_matrix1 = confusion_matrix(df["Category"], preds)

        cmd = ConfusionMatrixDisplay(confusion_matrix1, display_labels=model.classes_)

        df.reset_index(inplace=True, drop=True)

        return df, metrics_dict, cmd




# In[ ]:


class DataLoadDump:
    """
    This class helps in saving and loading objects in pickle files
    """
    def load_data(self,
                  path,
                  conn_string=None,
                  container_name=None,
                  in_blob_storage=None):
        """
        Returns the model object stored in the path
        """
        # FIXME - Mounting exposes data from specific workstream to entire Databricks Workspace. We use OAuth tokens to access ADLS including writing buffers direct to storage.

        if in_blob_storage == "YES":
            blob = BlobClient.from_connection_string(
                conn_string, container_name=container_name, blob_name=path)
            downloader = blob.download_blob(0)
            b = downloader.readall()
            loaded_model = pickle.loads(b)

            return loaded_model

        else:
            with open(path, "rb") as input_file:
                model = pickle.load(input_file)

            return model

    def dump_data(
        self,
        path,
        model_object,
        conn_string=None,
        container_name=None,
        in_blob_storage=None,
    ):
        """
        Saves the object in a particular location

        input
        ------
        :param path (str): Path where the object has to be stored
        :param moel_object (object): Python object that has to be stored

        Return
        -------
        True if object saved

        """
        # FIXME - Mounting exposes data from specific workstream to entire Databricks Workspace. We use OAuth tokens to access ADLS including writing buffers direct to storage.

        if in_blob_storage == "YES":
            blob = BlobClient.from_connection_string(
                conn_string, container_name=container_name, blob_name=path)

            pickle_file = pickle.dumps(model_object)

            blob.upload_blob(pickle_file)

            return True

        else:
            with open(path, "wb") as output_file:
                pickle.dump(model_object, output_file)

            return True

    def read_pandas_csv(self,
                        path,
                        conn_string=None,
                        container_name=None,
                        in_blob_storage=None):
        df = pd.read_csv(f"abfs://" + container_name + "/" + path,
                         storage_options={"connection_string": conn_string})
        return df

    def write_pandas_csv(self,
                         path,
                         df,
                         conn_string=None,
                         container_name=None,
                         in_blob_storage=None):
        df.to_csv(
            f"abfs://" + container_name + "/" + path,
            storage_options={"connection_string": conn_string},
        )
        return df


# In[ ]:


class ExtractRawData:

    #Function to convert the raw_data in url form to fields required for inferencing
    def get_forms_data(self, activity_id, asset_name, raw_data, form_map_dict):
        fields_ext = urllib.parse.parse_qsl(
            raw_data,
            strict_parsing=False,
            keep_blank_values=True,
            errors="replace",
            encoding="utf-8",
            max_num_fields=None,
            separator="&",
        )
        extracted_field_dict = dict(fields_ext)
        sub_dict = {}
        temp_dict = form_map_dict.get(asset_name)
        if temp_dict != None:
            for field in extracted_field_dict.keys():
                mapped_field = {i for i in temp_dict if temp_dict[i] == field}
                if mapped_field != set():
                    (new_col_name, ) = mapped_field
                    sub_dict.update(
                        {new_col_name: extracted_field_dict[field].strip()})
                else:
                    pass
            sub_dict.update({"business_line": temp_dict.get('business_line')})
        else:
            activity_id = ""

        sub_dict.update({"activity_id": str(activity_id)})
        return sub_dict


# In[ ]:





# In[ ]:




