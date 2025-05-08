import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, dendrogram




st.set_page_config(page_title="Aidan Finn Final Project", layout="wide")
st.title("Un-supervised Machine Learning App")
st.subheader("How to work the app")
st.write("This app is a demonstration of the potential of two machine learning techniques - K-Means Clustering and Hierarchical Clustering - to reveal structure in unlabeled data." \
"\nYou can choose between the two techniques. If you choose K-Means, you can see an analysis of a pre-uploaded dataset or of one you upload; if you choose Hierarchical Clustering, you can see an analysis of a different pre-uploaded dataset.")
#Functions

#Cleaning Datasets
#Loading Datasets
def load_datasets(dataset, selected_dataset):
    #Choose or Upload/process dataset
    if selected_dataset == "pre-uploaded":
        df = dataset
        X = df.data
        y = df.target
        features = df.feature_names
        targets = df.target_names
        return df, X, y, features, targets
    else:
        if dataset is not None:
            df = pd.read_csv(dataset)
            feature_input = st.text_input("Enter the features from your dataset (separated by commas): ")
            #Process dataset
            features = [feature.strip() for feature in feature_input.split(',')] #Originally I just had the last part in brackets; then I figured out that the brackets were causing a problem; then I did some more research anded added the iterated strip function since it's natural to add spaces
            features = [feature for feature in features if feature != ""] #after much googling (and that really search labs AI overview that pops up at the top of google searches) I found this idea
            target_input = st.text_input("Enter the target from your dataset: ")
            targets = [target.strip() for target in target_input.split(',')]
            #Clean dataset
            X = df[features].dropna().drop_duplicates()
            if targets:
                y = df[targets]
                return df, X, y, features, targets
            else:
                return None, None, None, None
        else:
            return None, None, None, None #Prevents error message from occuring before user has uploaded a dataset
        
#Split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

#Scale data
def scale_data(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std
    
#KMeans Clustering
def plot_optimal_k(X_std):
    # Define the range of k values to try
    ks = range(2, 11)  # starting from 2 clusters to 10 clusters
    wcss = []               # Within-Cluster Sum of Squares for each k
    silhouette_scores = []  # Silhouette scores for each k
    # Loop over the range of k values
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_std)
        wcss.append(km.inertia_)  # inertia: sum of squared distances within clusters
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))   
    # Explain graphs
    st.subheader("What is KMeans Clustering?")
    st.text("KMeans Clustering groups an un-labeled dataset into different clusters. These clusters can represent labels. By comparing the clusters to the known labels, we can assess the performance of the model." \
    "\nFor example, a KMeans clustering model can be trained to identify cars vs trucks if we give it unlabeled vehicles with variables like size, mpg, and engine displacement." \
    "\nThe model will cluster the unlabeled vehicles. If we have the labels, we can compare the model's clusters with actual labels. If they match well, we can use the model to make future predictions.")
    st.subheader("\nChoosing the optimal number of clusters (if you wish to predict a set number fo labels, choose that number)")
    st.text("Below, you will see the an 'Elbow' Plot and a Silhouette Score Plot.\n" \
    "To find the optimal number of clusters, look at THE HIGHEST SILHOUETTE SCORE and the point RIGHT BEFORE THE ELBOW PLOT DROPS (i.e. bends like an elbow).")
    st.subheader("\nAbout the pre-uploaded wine dataset")
    st.text("The wine dataset is a collection of data from three Italian wine-producing regions (named 'class_0,' 'class_1,' 'class_2). Researchers measured several variables from over a hundred different observations." \
    "\n We can remove the labels (Wine Region) and use the unlabeled data to use the variables from the observations to predict the region of each observation.")
    # Plot the Elbow Method result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, wcss, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)

    # Plot the Silhouette Score result
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette_scores, marker='o', color='green')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

def select_optimal_k():
    k = st.select_slider("Select 'k' value:", options=list(range(1, 11)))
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_std)
    return clusters, k

def display_KM_scatterplot(clusters, k, target_names):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    colors = ['red','blue','yellow','lightgreen','orange','purple','black','pink','darkgreen','darkred','darkorange']
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    for i in range(k):
        plt.scatter(X_pca[clusters == i, 0,], X_pca[clusters == i, 1],
                c=colors[i], alpha=0.7, edgecolor='k', s=60, label=f"Cluster {i+1}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D Grouping Prediction')
    plt.legend(loc='best')
    plt.grid(True)
   
    #Compare to known labels
    plt.subplot(1, 2, 2)
    for j, target_name in enumerate(target_names):
        plt.scatter(X_pca[y == j, 0], X_pca[y == j, 1],
                    color=colors[j], alpha=0.7, edgecolor='k', s=60, label=target_name)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('True Labels')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

#Hierarchical Clustering
def explain_dendrogram():
    st.subheader("What is hierarchical clustering")
    st.write("Hierarchical clustering is a form of machine learning that helps us identify patterns and multi-leveled structures in unlabeled data." \
    "\n Hierarchical clustering can also help us separate large datasets into smaller groups for further analysis." \
    "\n To do this, we use a DENDROGRAM - a type of visualization that resembles an upside down tree, with the 'leaves' (individual observations) at the bottom.")
    st.subheader("What is this particular dataset?")
    st.write("This dataset is a list of observations of countries and attached variables. " \
    "\nI chose this dataset over the 'wine' dataset I use in KMeans Clustering because it is better represented by a dendrogram.")
def plot_dendrogram(df):
    features_df = df.drop(columns = "country")
    X_scaled = scale_data(features_df)
    country = df["country"].to_list()


    Z = linkage(X_scaled, method="ward")      # linkage matrix
    plt.figure(figsize=(20, 7))
    dendrogram(Z,
           #truncate_mode="lastp", # truncates the number of examples shown, comment out to see all countries
           labels = country)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Targets")
    plt.ylabel("Distance")
    st.pyplot(plt)
    plt.clf()

#main code
dataset_options = ["pre-uploaded", "upload my own"]
unsupervised_options = ["KMeans clustering", "Hierarchical Clustering"]
unsupervised_selected = st.selectbox("Selected Unsupervised ML Model:", unsupervised_options)

if unsupervised_selected == "KMeans clustering":
    selected_dataset = st.selectbox("Select dataset:", dataset_options)
    if selected_dataset == "upload my own":
        dataset = st.file_uploader("Upload a pre-cleaned dataset as a CSV file:")
    else:
        dataset = load_wine()

    df, X, y, features, targets = load_datasets(dataset, selected_dataset)
    split_data(X, y, test_size=0.2, random_state=42)
    X_std = scale_data(X)
    plot_optimal_k(X_std)
    clusters, k = select_optimal_k()
    display_KM_scatterplot(clusters, k, targets)

else:
    df = pd.read_csv("C:/Users/amcco/Downloads/Country-data.csv")
    explain_dendrogram()
    plot_dendrogram(df)

#References
#https://docs.streamlit.io/develop/api-reference/widgets/st.select_slider
