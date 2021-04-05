"""
script for baseline model
transform text to vectors using TD-IDF
prediction using log regression

Returns:
    float: accuracy score
"""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.datapipeline import PROCESSED_DATA_PATH

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("baseline_model")


@dataclass
class BaselineModel:
    """
    extract text features using tdidf
    predict sentiment using log regression
    """

    df_train: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_train.csv")
    df_val: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_val.csv")
    df_test: pd.DataFrame = pd.read_csv(PROCESSED_DATA_PATH / "df_test.csv")
    model: Pipeline = Pipeline(
        [("vectorise", TfidfVectorizer()), ("lr", LogisticRegression())]
    )

    def train(self) -> None:
        """Fitting model to training set"""
        logger.info("fitting model")
        self.model.fit(self.df_train["cleaned_text"], self.df_train["label"])

    def evaluate(self) -> float:
        """Evalute model on test set

        Returns:
            float: accuracy score
        """

        logger.info("evaluating model")
        train_acc = self.model.score(
            self.df_train["cleaned_text"], self.df_train["label"]
        )
        test_acc = self.model.score(self.df_test["cleaned_text"], self.df_test["label"])
        return train_acc, test_acc


if __name__ == "__main__":
    base = BaselineModel()
    base.train()
    train_acc, test_acc = base.evaluate()
    logger.info("train accuracy: %f; test accuracy: %f", train_acc, test_acc)
