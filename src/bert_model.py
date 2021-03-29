from dataclasses import dataclass

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from transformers import (BertTokenizer, InputExample, InputFeatures,
                          TFBertForSequenceClassification)

from src.datapipeline import PROCESSED_DATA_PATH

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


@dataclass
class BertModel:
    df_train: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_train.csv")
    df_val: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_val.csv")
    df_test: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_test.csv")

    def build(self):
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.summary()
        return model

    def convert_data_to_examples(
        self, train, val, test, DATA_COLUMN: str = "text", LABEL_COLUMN: str = "text"
    ):
        """take in train, test and val dataset and convert each row into an InputExample object"""
        train_InputExamples = train.apply(
            lambda x: InputExample(
                guid=None,  # Globally unique ID for bookkeeping, unused in this case
                text_a=x[DATA_COLUMN],
                text_b=None,
                label=x[LABEL_COLUMN],
            ),
            axis=1,
        )

        validation_InputExamples = test.apply(
            lambda x: InputExample(
                guid=None, text_a=x[DATA_COLUMN], text_b=None, label=x[LABEL_COLUMN]
            ),
            axis=1,
        )
        test_InputExamples = test.apply(
            lambda x: InputExample(
                guid=None, text_a=x[DATA_COLUMN], text_b=None, label=x[LABEL_COLUMN]
            ),
            axis=1,
        )

        return train_InputExamples, validation_InputExamples, test_InputExamples
