import json
import os
import random
import networkx
import numpy as np
import glob
import pathlib
import nltk
import re
import string
import pandas as pd
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from gensim.summarization.summarizer import summarize

def chooseFiles(directory, percentage):
    input = []
    input.extend(glob.glob(directory, recursive=True))
    jsonFiles = [i for i in random.sample(input, round(len(input) * (percentage / 100)))]
    print(len(jsonFiles))
    # print(input)
    documentData = []
    for i in jsonFiles:
        with open(i, 'r') as f:
            json_data = json.load(f)
        body_text = json_data['body_text']
        fileData = ""
        for text in body_text:
            fileData += text['text']
        documentData.append(fileData)
    print("Choose File Method Completed")
    return documentData

def tokenization(data, sentTokenize = False):
    exclude = set(string.punctuation)
    tokenData = []
    regex =  r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*(?<![.,])'
    if sentTokenize == True:
        tempData = str(data)
        text = re.sub(regex, '', tempData)
        tokenData = nltk.sent_tokenize(tempData)
    else:
        for list in data:
            list = list.lower()
            text = re.sub(regex, '', list)
            token = nltk.word_tokenize(text)
            tokenWord = [i for i in token if i not in stopwords.words("english") and i not in exclude ]#and i not in exclude
            mytokens = " ".join([word for word in tokenWord])
            tokenData.append(mytokens)
    print("Tokenization Completed")
    return tokenData

def Vectorization(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    vector = vectorizer.fit_transform(data)
    # df = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())
    print(vector.toarray())
    print(vector.shape)
    print("Vectorization Completed")
    return vector

def optimalCluster(vector):
    pca = PCA(n_components=0.95, random_state=42)
    pcaReducedDim = pca.fit_transform(vector.toarray())
    # print(pcaReducedDim)
    # print(pcaReducedDim.shape)
    # print(vector.shape[0])
    distortions = []
    K = range(2, 10, 2)
    for k in K:
        k_means = KMeans(n_clusters=k, random_state=10).fit(pcaReducedDim)
        k_means.fit(pcaReducedDim)
        distortions.append(sum(np.min(cdist(pcaReducedDim, k_means.cluster_centers_, 'euclidean'), axis=1)) / vector.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'b-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    print("Optimal Cluster Completed")
    return pcaReducedDim

def clustering(matrix, actualData):
    clusterData = []
    #KMean Cluster
    optCluster = 10
    model = KMeans(n_clusters=optCluster, init='k-means++', n_init=10, max_iter=250,verbose=0, random_state=None, precompute_distances='auto',copy_x=True, n_jobs=None, algorithm='auto')
    model.fit(matrix)
    # print(model.labels_)
    # print(model.cluster_centers_)
    # print(silhouette_score(matrix,model.labels_))

    #Preparing Clustered Data
    for centers in range(len(model.cluster_centers_)):
        data = []
        for labels in range(len(model.labels_)):
            if centers == model.labels_[labels]:
                data.append(actualData[labels])
        clusterData.append(data)
    # print(clusterData)
    for i in range(len(clusterData)):
        print(len((clusterData[i])))
    # print(clusterData)
    print("Cluster Completed")
    return clusterData

def summarization(data):
    for i in range(len(data)):
        strData = "".join(data[i])
        # print("String Data:" ,i, strData)
        summarized = summarize(strData, ratio=1, split=True)[0:10]
        # print("Summary Length:",len(summarized))
        # print("summarized Data:",summarized)
        location = os.curdir + "/outputFiles"
        summarizedData = open(os.path.join(location, 'Cluster' + str(i)), 'w', encoding="utf-8")
        for j in range(len(summarized)):
            summarizedData.write(summarized[j])
            summarizedData.write('\n')
        summarizedData.close()
    print("Summarization Completed")



directory = "D:/Spring Semester/Text Analytics/Project2/CORD-19-research-challenge/**/*.json"
data = chooseFiles(directory, percentage=5)
tokenizedDate = tokenization(data)
X = Vectorization(tokenizedDate)
pcaMatrix = optimalCluster(X)
clusterData = clustering(pcaMatrix, data)
summarization(clusterData)
