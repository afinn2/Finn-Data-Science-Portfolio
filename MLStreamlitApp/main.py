import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





st.title("Supervised Machine Learning App")

#Functions

#Loading Datasets
dataset_options = ["mpg", "titanic", "upload my own"]
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
    if dataset == "titanic":
        df.dropna(subset=['age'], inplace=True)
        df = pd.get_dummies(df, columns=['sex'], drop_first=True)
        features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
        X = df[features]
        y = df['survived']
        return df, X, y, features
    elif dataset == "mpg":
        df.dropna(subset=['horsepower'], inplace=True)
        features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
        X = df[features]
        y = df['mpg']
        return df, X, y, features
    else:
        feature_input = st.text_input("Enter the features from your dataset (separated by commas): ")
        #Split by commas to create feature list
        features = [feature.strip() for feature in feature_input.split(',')] #Originally I just had the last part in brackets; then I figured out that the brackets were causing a problem; then I did some more research anded added the iterated strip function since it's natural to add spaces
        features = [feature for feature in features if feature != ""] #after much googling (and that really search labs AI overview that pops up at the top of google searches) I found this idea
        target = st.text_input("Enter the target from your dataset: ")
        X = df[features]
        if target:
            y = df[target]
            return df, X, y, features
        else:
            return None, None, None, None
#Split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#Train KNN
def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

#Confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    plt.clf()

def knn(X, y):
    #_______________________________KNN___________________________________________________________
    # Selection controls at the top
    k = st.slider("Select number of neighbors (k, odd values only)", min_value=1, max_value=21, step=2, value=5)
    st.header("Information and Visualizations")

    #Load, preprocess, and scale the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train KNN with the selected k value
    knn_model = train_knn(X_train, y_train, n_neighbors=k)
    st.write(f"**Scaled Data: KNN (k = {k})**")

    # Predict and evaluate
    y_pred = knn_model.predict(X_test)
    accuracy_val = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy at predicting survivorship:")
    st.write(f"**{accuracy_val:.2f}**")

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, f"KNN Confusion Matrix - Scaled Data")

    with col2:
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

    #___________________________ROC and AUC score_______________________________
    # Get the predicted probabilities for the positive class (survival)
    y_probs = knn_model.predict_proba(X_test)[:, 1]

    # Calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    # Compute the Area Under the Curve (AUC) score
    roc_auc = roc_auc_score(y_test, y_probs)

    # Plot the ROC curve
    st.subheader("ROC Curve")
    plot = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='50/50 Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plot)
    
def linear_regression(X, y):    
    # Initialize and train logistic regression model
    model = LinearRegression()
    model.fit(X, y)

    #Evaluate Model
    # Predict on test data
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Calculate performance
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)#I was having import problems with RMSE so I calculated manually since it's super easy
    r2 = r2_score(y, y_pred)
    st.subheader("(R)MSE and R^2: How well does model predict MPG")
    st.write(f"RMSE: {rmse:.3f}")
    st.write(f"MSE: {mse:.3f}")
    st.write(f"R^2: {r2:.3f}")

    #Assess performance (residuals)
    st.subheader("Visualization of Performance")
    st.write("A bell curve with residuals centered around '0' indicates good performance.")
    residual_hist = plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    st.pyplot(residual_hist)

#main code
selected_dataset = st.selectbox("Select dataset:", dataset_options)
if selected_dataset == "mpg":
    df, X, y, features = load_datasets("mpg") #Regression Model
    linear_regression(X, y)
elif selected_dataset == "titanic":
    df, X, y, features = load_datasets("titanic") #Classification model - KNN
    knn(X, y)
else:
    uploaded_dataset = st.file_uploader("Upload a pre-cleaned dataset as a CSV file:")
    df, X, y, features = load_datasets(uploaded_dataset)
    if df is not None:
        prediction_category = st.selectbox("What kind of prediction do you want: ", ["Classification", "Regression"])
        if uploaded_dataset != None:
            if features and y is not None:
                if prediction_category == "Classification":
                    knn(X, y)
                else:
                    linear_regression(X, y)



