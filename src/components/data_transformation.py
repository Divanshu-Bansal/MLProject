import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Reading the training data")
            train = pd.read_csv("artifacts/train.csv")

            categorical_columns = [feature for feature in train.columns if train[feature].dtype== 'str']
            numerical_columns = [feature for feature in train.columns if train[feature].dtype != 'str']
            numerical_columns.remove('math score')

            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("one hot encoder", OneHotEncoder())
                ]
            )

            logging.info("Numerical features scaling completed")
            logging.info("Catrgorical features encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ['reading score', 'writing score']
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocesssing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )    

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            