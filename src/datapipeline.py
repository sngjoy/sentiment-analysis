"""
Reading in and cleaning data files
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

logging.basicConfig(
    format="[%(asctime)s] %(levelname)-8s %(name)-15s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("datapipeline")

SAVE = True
OVERWRITE = False

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

    df: pd.DataFrame = None

    @classmethod
    def load(cls, data_path: Union[Path, str] = RAW_DATA_PATH) -> pd.DataFrame:
        """
        load raw text
        clean text

        Args:
            data_path (Union[Path, str], optional): path to data folder. Defaults to RAW_DATA_PATH.

        Returns:
            pd.DataFrame: data
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

        logger.info("cleaning text")
        df["cleaned_text"] = df["text"].apply(cls.clean_text)

        return cls(df=df)

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

    if output_data_path.exists() and not OVERWRITE:
        logger.info(
            "%s exist, please load it from filesystem",
            output_data_path.name,
        )
    else:
        pipe = Datapipeline.load()

        if SAVE:
            pipe.save(pipe.df, PROCESSED_DATA_PATH, "data.csv")
