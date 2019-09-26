# Intrend - into Google Trends

## Introduction
Using Google Trends to develop some simple machine learning models and comparing them to Facebook Prophet

## Content
- A simple estimation model based on 3 different kinds of networks, made using Keras and Tensorflow
- An analysis of the same data using Prophet by Facebook
- A set of data ready to be used in both models and fetched manually from the Google Trends site

## About the models
The chosen models are CNN LSTM and MLP: the value of the parameters should be optimized, but experimenting is the best way to improve the predictions. The important parameters are:
- Number of epochs
- Early stopping condition
- Training and validation set size
- Prediction window (predicted_months)
- Tensors size (observation_size)

## About the data
The data folder contains 4 sub-folders:
- 1-easy: keywords with a strong and regular periodic pattern (example: "halloween")
- 2-medium: keywords with a strong but unregular periodic pattern (example: "black friday")
- 3-hard: keywords with an unregular and not periodic pattern (example: "avengers")
- 4-no-pattern: keywords with no patterns at all (example: "microsoft")

The name of the file contains the keyword and the geographic zone (for example, wr stays for world, it for Italy)

## Useful resources
https://trends.google.com/trends/story/US_cu_6fXtAFIBAABWdM_en

https://github.com/facebook/prophet

https://medium.com/@pewresearch/using-google-trends-data-for-research-here-are-6-questions-to-ask-a7097f5fb526