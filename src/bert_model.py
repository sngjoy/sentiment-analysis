from dataclasses import dataclass
from typing import List

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
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        DATA_COLUMN: str = "text",
        LABEL_COLUMN: str = "text",
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

    def convert_examples_to_tf_dataset(
        examples: List[InputExample], tokenizer: BertTokenizer, max_length: int = 128
    ):
        features = []

        for e in examples:
            input_dict = tokenizer.encode_plus(
                e.text_a,
                add_special_tokens=True,
                max_length=max_length,  # truncates if len(s) > max_length
                return_token_type_ids=True,
                return_attention_mask=True,
                pad_to_max_length=True,  # pads to the right by default
                truncation=True,
            )

            input_ids, token_type_ids, attention_mask = (
                input_dict["input_ids"],
                input_dict["token_type_ids"],
                input_dict["attention_mask"],
            )

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=e.label,
                )
            )

        def gen():
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "token_type_ids": tf.int32,
                },
                tf.int64,
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )
