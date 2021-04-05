"""
script for LSTM model

Returns:
    float: accuracy score of train, val and test
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.datapipeline import PROCESSED_DATA_PATH

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("LSTM model")


@dataclass
class LSTMModel:

    df_train: pd.DataFrame = pd.read_csv(
        PROCESSED_DATA_PATH / "df_train.csv", usecols=["cleaned_text", "label"]
    )[:200]
    df_val: pd.DataFrame = pd.read_csv(
        PROCESSED_DATA_PATH / "df_val.csv", usecols=["cleaned_text", "label"]
    )[:100]
    df_test: pd.DataFrame = pd.read_csv(
        PROCESSED_DATA_PATH / "df_test.csv", usecols=["cleaned_text", "label"]
    )[:100]
    train_pad: np.ndarray = None
    val_pad: np.ndarray = None
    test_pad: np.ndarray = None
    vocab_size: int = None
    max_length: int = None

    def tokenise(self) -> None:
        """
        Vectorise the text corpus
        associate each word to a number
        transform text into a sequence of numbers
        """
        logger.info("extracting word embeddings")

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(
            self.df_train["cleaned_text"]
        )  # creates a vocabulary index based on word frequency
        max_length = 500

        # transforms each text in texts to a sequence of integers.
        train_token = tokenizer.texts_to_sequences(self.df_train["cleaned_text"])
        val_token = tokenizer.texts_to_sequences(self.df_val["cleaned_text"])
        test_token = tokenizer.texts_to_sequences(self.df_test["cleaned_text"])

        # padding to make all the vectors of same length
        self.train_pad = pad_sequences(train_token, maxlen=max_length, padding="post")
        self.val_pad = pad_sequences(val_token, maxlen=max_length, padding="post")
        self.test_pad = pad_sequences(test_token, maxlen=max_length, padding="post")

        # params that are needed for embedding layer of model
        self.vocab_size = len(tokenizer.word_index) + 1
        self.max_length = max_length

    def build_model(self) -> None:
        """
        Build LSTM model
        """
        logger.info("building model")
        model = Sequential(
            [
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=32,
                    input_length=self.max_length,
                ),
                LSTM(
                    units=64,
                    dropout=0.2,  # units to drop for the linear trf of input
                    recurrent_dropout=0.2,  # units to drop for the linear trf of recurrent state
                ),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model

    def train(self, epochs: int, batch_size: int = 128) -> None:
        """
        fit LSTM model to train set

        Args:
            epochs (int): number of epochs
            batch_size (int, optional): batch size. Defaults to 128.
        """

        logger.info("fitting model")
        self.model.fit(
            self.train_pad,
            self.df_train["label"],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.val_pad, self.df_val["label"]),
        )

    def evaluate(self) -> float:
        """
        evaluate accuracy of model

        Returns:
            float: accuracies of train, val and test sets
        """
        logger.info("evaluating model")
        _, train_acc = self.model.evaluate(self.train_pad, self.df_train["label"])
        _, val_acc = self.model.evaluate(self.val_pad, self.df_val["label"])
        _, test_acc = self.model.evaluate(self.test_pad, self.df_test["label"])
        return train_acc, val_acc, test_acc


if __name__ == "__main__":
    lstm = LSTMModel()
    lstm.tokenise()
    lstm.build_model()
    lstm.train(epochs=5, batch_size=32)
    train_acc, val_acc, test_acc = lstm.evaluate()
    logger.info(
        "Train accuracy: %f; Val accuracy: %f; Test accuracy: %f",
        train_acc,
        val_acc,
        test_acc,
    )
