# sentiment_analysis

WIP

### Introduction
This project aims to predict the sentiment of movie reviews. The dataset used is from Stanford University which can be downloaded from this [link](https://ai.stanford.edu/~amaas/data/sentiment/).

### Datapipeline
`src/datapipeline.py`

Cleaning text, splitting them into train, val and test sets (80-10-10)

### Modelling
1. Baseline model `src/baseline_model.py`

    - Text is converted into a vector representation using TD-IDF and logistic regression is used to predict the labels. 
    - Train accuracy: 0.932 
    - Test accuracy: 0.896

2. LSTM model `src/lstm_model.py`

    - Train accuracy: 0.997
    - Validation accuracy: 0.862
    - Test accuracy: 0.861
