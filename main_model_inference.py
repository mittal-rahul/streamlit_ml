#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import requests, json
import logging

from common_function_library import *


logger = logging.getLogger(__name__)


class MainModelInference:
    """
    Class to predict the pred_label, probability, calculated_label, score and priority for unseen data.
    """
    def main_model_predict(
        self,
        nlp,
        sw_spacy,
        df,
        model,
        vectorized_vocab_path,
        threshold,
        conn_string,
        container_name,
        in_blob_storage=None,
    ):
        """ "
        Predicts the probability, pred_label and calculated_label of the input dataframe.
        :param df(DataFrame): Input dataframe with ngrams
        :param model(LogisticRegression()): Trained logistic regression model
        :param vectorized_vocab (Dictionary): Trained and vectorized Tf-Idf vectorizer model vocabulary
        :param threshold (int): Threshold probability of predictions below which
                                the predictions will be marked as "Uncertain"
        :return df(DataFrame): Returns input dataframe with added columns "probability",
                                "pred_label" and "calculated_label"

        """

        data_transform = DataTransformation(nlp, sw_spacy)
        try:
            df = data_transform.transform_data(df)
        except Exception as e:
            logger.error("Error occured in data transformation", exc_info=e)
            raise e

        data_load_dump = DataLoadDump()
        model = data_load_dump.load_data(model, conn_string, container_name,
                                         in_blob_storage)
        if type(model) != False:
            logger.info("ML Model loaded")
        else:
            logger.error("ML Model could not be loaded")
            raise Exception("ML model could not be loaded")
        vectorized_vocab = data_load_dump.load_data(vectorized_vocab_path,
                                                    conn_string,
                                                    container_name,
                                                    in_blob_storage)
        if type(vectorized_vocab) != False:
            logger.info("Vocab loaded")
        else:
            logger.error("Vocab could not be loaded")
            raise Exception("Vocab could not be loaded")

        # Fitting and transforming the data based on pre-trained vocabulary
        vect = TFIDFVectorizer()
        vectorizer = vect.corpus_vectorizer(vectorized_vocab)
        try:
            corpus_vectors = vectorizer.fit_transform(df["ngrams"])
        except Exception as e:
            logger.error("Error in vectorizing the data", exc_info=e)
        preds = model.predict(corpus_vectors)
        df["pred_label"] = preds
        preds_proba = model.predict_proba(corpus_vectors)

        # Initializing object of class CustomLabelling() to access label_calculated() function
        custom_label = CustomLabelling()

        df["probability"] = np.amax(preds_proba, axis=1)
        df["calculated_label"] = df.apply(
            lambda x: custom_label.label_calculated(x["pred_label"], x[
                "probability"], threshold),
            axis=1,
        )

        return df

    def ranking_model(
        self,
        score_file,
        param_imp_file,
        profile_map_file,
        keywords_category_file,
        df,
        description_present,
        folder_path,
        conn_string,
        container_name,
        in_blob_storage,
    ):
        """ "
        Function to check the score and priority of the enquiries.
        :param df(DataFrame): Input Dataframe containing colums "existing_customer","calculated_label",
                              "translated_profile","ngrams"
        :param description_present(boolean): Input value 1 for enquiries with
                                              description and 0 for enquiries
                                              without description
        :param folder_path(str): path of the Inquiry Ranking Mapping file where all the mappings are defined
        :return df(DataFrame): Returns input dataframe with added columns "score",
                                "priority" and "existing_customer" transformed to
                                "Existing" for True and "New" for False

        """
        # Initializing ScoreCalculation class to use its calculate_score and
        # calculate_priority functions for getting the score and priority of the enquiries

        df.fillna("", inplace=True)
        df["existing_customer"] = df["existing_customer"].replace(
            [True, False], ["Existing", "New"])
        try:
            ranking = ScoreCalculation(score_file, param_imp_file,
                                       profile_map_file,
                                       keywords_category_file, folder_path,
                                       conn_string, container_name,
                                       in_blob_storage)
            df["score"] = df.apply(
                lambda x: ranking.calculate_score(x, description_present),
                axis=1)
            df["priority"] = df["score"].apply(
                lambda x: ranking.calculate_priority(x))

            return df
        except Exception as e:
            logger.error("Error in assigning priority", exc_info=e)
            raise e

