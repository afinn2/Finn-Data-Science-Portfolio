
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt






st.title("Un-supervised Machine Learning App")

#Functions

#Cleaning Datasets
#Loading Datasets
dataset_options = []
def load_datasets(dataset):
    #Choose or Upload dataset
    if selected_dataset == "upload my own":
        if uploaded_dataset is not None:
            df = pd.read_csv(dataset)
        else:
            return None, None, None, None #Prevents error message from occuring before user has uploaded a dataset
    else:
        df = sns.load_dataset(dataset)
    #Process Selected Dataset
    # if dataset == "titanic":
    #     df.dropna(subset=['age'], inplace=True)
    #     df = pd.get_dummies(df, columns=['sex'], drop_first=True)
    #     features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
    #     X = df[features]
    #     y = df['survived']
    #     return df, X, y, features
    # elif dataset == "mpg":
    #     df.dropna(subset=['horsepower'], inplace=True)
    #     features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
    #     X = df[features]
    #     y = df['mpg']
    #     return df, X, y, features
    # else:
    #     feature_input = st.text_input("Enter the features from your dataset (separated by commas): ")
    #     #Split by commas to create feature list
    #     features = [feature.strip() for feature in feature_input.split(',')] #Originally I just had the last part in brackets; then I figured out that the brackets were causing a problem; then I did some more research anded added the iterated strip function since it's natural to add spaces
    #     features = [feature for feature in features if feature != ""] #after much googling (and that really search labs AI overview that pops up at the top of google searches) I found this idea
    #     target = st.text_input("Enter the target from your dataset: ")
    #     X = df[features]
    #     if target:
    #         y = df[target]
    #         return df, X, y, features
    #     else:
    #         return None, None, None, None
#Split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#Confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()



#main code
selected_dataset = st.selectbox("Select dataset:", dataset_options)
# if selected_dataset == "mpg":
#     df, X, y, features = load_datasets("mpg") #Regression Model
#     linear_regression(X, y)
# elif selected_dataset == "titanic":
#     df, X, y, features = load_datasets("titanic") #Classification model - KNN
#     knn(X, y)
# else:
#     uploaded_dataset = st.file_uploader("Upload a pre-cleaned dataset as a CSV file:")
#     df, X, y, features = load_datasets(uploaded_dataset)
#     if df is not None:
#         prediction_category = st.selectbox("What kind of prediction do you want: ", ["Classification", "Regression"])
#         if uploaded_dataset != None:
#             if features and y is not None:
#                 if prediction_category == "Classification":
#                     knn(X, y)
#                 else:
#                     linear_regression(X, y)
