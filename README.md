# sentiment_analysis

WIP

### Introduction
This project aims to predict the sentiment of movie reviews. The dataset used is from Stanford University which can be downloaded from this [link](https://ai.stanford.edu/~amaas/data/sentiment/)

### Datapipeline
`src/datapipeline.py`

Cleaning text and splitting them into train, val and test sets.

### Modelling
1. Baseline model `src/baseline_model.py`

    - Text is converted into a vector representation using TD-IDF and logistic regressing is used to predict the labels. 
    - Accuracy: 0.903 