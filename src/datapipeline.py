"""
Reading in and cleaning data files
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("datapipeline")

SAVE = True
OVERWRITE = True

DATA_PATH = Path(".").absolute() / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# pylint: disable=invalid-name
@dataclass
class Datapipeline:
    """
    Load, preprocess and save text data
    To load and save data:
        >>> pipe = Datapipeline.load(data_path)
        >>> pipe = pipe.save(pipe.df, output_file_path, output_filename)
    """

    df_train: pd.DataFrame = None
    df_val: pd.DataFrame = None
    df_test: pd.DataFrame = None

    @classmethod
    def load(cls, data_path: Union[Path, str] = RAW_DATA_PATH) -> pd.DataFrame:
        """
        load raw text
        clean text
        split data into train, val, test sets

        Args:
            data_path (Union[Path, str], optional): path to data folder. Defaults to RAW_DATA_PATH.

        Returns:
            pd.DataFrame: train, val, test dataset
        """
        logger.info("loading raw data")
        text = []
        label = []

        for folder in data_path.glob("*"):
            for folder2 in folder.glob("*"):
                if folder2.name == "pos":
                    for textfile in folder2.glob("*.txt"):
                        with open(textfile, "rb") as f:
                            data = f.read()
                            data = data.decode("utf-8", "ignore")
                        text.append(data)
                        label.append(folder2.name)

                elif folder2.name == "neg":
                    for textfile in folder2.glob("*.txt"):
                        with open(textfile, "rb") as f:
                            data = f.read()
                            data = data.decode("utf-8", "ignore")
                        text.append(data)
                        label.append(folder2.name)

        df = pd.DataFrame({"text": text, "label": label})

        df = df.drop_duplicates(keep="first", ignore_index=True)

        df = cls.label_binariser(df, "label")

        logger.info("cleaning text")
        df["text"] = df["text"].apply(cls.remove_html)
        df["cleaned_text"] = df["text"].apply(cls.clean_text)
        df_train, df_val, df_test = cls.train_test_val_split(df)

        return cls(df_train=df_train, df_val=df_val, df_test=df_test)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        lowercase, decontract text
        remove special characters
        remove extra while space

        Args:
            text (str): raw text

        Returns:
            str: cleaned text
        """
        text = text.lower()

        # decontracting words
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)

        text = re.sub(
            r"[^a-zA-Z0-9]", " ", text
        )  # remove all non-alphanumeric characters
        text = re.sub(r"\s+", " ", text)  # remove extra white spaces

        return text

    @staticmethod
    def remove_html(text: str) -> str:
        """
        remove html artifacts from text

        Args:
            text (str): text in data

        Returns:
            str: text without html formatting
        """

        soup = BeautifulSoup(text, "lxml")
        return soup.text

    @staticmethod
    def label_binariser(df: pd.DataFrame, columns: Union[List, str]) -> pd.DataFrame:
        """
        change label into 1 (pos) and 0 (neg)

        Args:
            df (pd.DataFrame): dataset
            columns (Union[List, str]): column names to be binarised

        Returns:
            pd.DataFrame: dataset with binarised dataset
        """
        logger.info("binarising label")
        lb = LabelBinarizer()
        df[columns] = lb.fit_transform(df[columns])
        return df

    @staticmethod
    def train_test_val_split(df: pd.DataFrame) -> pd.DataFrame:
        """
        perform splitting of data into train, val, test

        Args:
            df (pd.DataFrame): entire dataset

        Returns:
            pd.DataFrame: data in train, val, test sets
        """
        logger.info("splitting data")
        df_val, df_test = train_test_split(
            df, test_size=0.1, stratify=df["label"], random_state=18
        )
        df_train, df_val = train_test_split(
            df_val, test_size=0.1, stratify=df_val["label"], random_state=18
        )

        return df_train, df_val, df_test

    @staticmethod
    def save(data: pd.DataFrame, output_file_path: Path, output_filename: str) -> None:
        """
        save data as a csv file

        Args:
            data (pd.DataFrame): data
            output_file_path (Path): output data path
            output_filename (str): output file name
        """

        output_file_path.mkdir(parents=True, exist_ok=True)

        data_path = output_file_path / f"{output_filename}.csv"

        data.to_csv(data_path, index=False)

        logger.info("%s saved", output_filename)


if __name__ == "__main__":
    logger.info(
        "Running %s, saving %s, overwrite %s",
        "datapipeline",
        SAVE,
        OVERWRITE,
    )

    output_data_path = PROCESSED_DATA_PATH / "data.csv"

    pipe = Datapipeline.load()

    if SAVE:
        for class_variable in list(vars(pipe).keys()):
            pipe.save(
                data=getattr(pipe, class_variable),
                output_file_path=PROCESSED_DATA_PATH,
                output_filename=f"{class_variable}",
            )
