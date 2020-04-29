## Text Summarization

Text summarization is the process of shortening a set of data computationally, to create a subset (a summary) that represents the most important or relevant information within the original content.Text summarization finds the most informative sentences in a document.

------
Packages Used:
----
- json
- os
- numpy
- glob
- nltk
- sklearn
- scipy
- matplotlib
- gensim

We use glob to read all the json file in a specific directory.
nltk package is used to tokenize word and sentences.
TfidfVectorizer, PCA decomposition, KMean clustering and find silhouette_score we use sklearn package.
matplotlib is used to plot the elbow curve.
gensim package is used for text summarization.

------
## summarization.py
----
The python file summarization.py file contains program for text summarization of all the documents. This program contains the following functions 
- chooseFiles()
- tokenization()
- Vectorization()
- optimalCluster()
- clustering()
- summarization()


### ChooseFiles():
The chooseFiles function takes 2 arguments which directory and percentage. The variable directory represents the directory in which the json documents are present. Percentage variable represents the percentage of files chosen randomly. In this function I have used glob module to finds all the pathnames matching a specified pattern and these document names are stored in a list. From the document list we select the random files based on the percentage variable. We then open each json document and extract the text data from the body_text element and store the text into a List.

### tokenization():
The tokenization function takes 2 arguments which are text data and a Boolean flag which defines whether the data must be sentence tokenize or word tokenized.
In this function we first remove any links present in the data, I have used a regular expression to remove the links. We than convert the data to lower case for convenience.
We then check for the Boolean variable, if it is true then we do a sentence tokenization.
If the variable is false we do word tokenization and also remove all the stop words by using nltk.corpus and also remove punctuations from the data.
This function returns tokenized data.

### Vectorization():
The vectorization function taken on 1 argument which is the tokenized data.
With the help of TfidfVectorizer from sklearn package, we vectorize the tokenized data to find the importance of a word or sentences in the document and across all the documents.
```
vectorizer = TfidfVectorizer(stop_words='english')
vector = vectorizer.fit_transform(data)
```
### optimalCluster():
The optimalCluster function taken in 1 argument which is vectorization matrix.
In this function we use Principal Component Analysis Technique to reduce the dimensionality of the vectorized matrix.
```
pca = PCA(n_components=0.95, random_state=42)
pcaReducedDim = pca.fit_transform(vector.toarray())
```
In this method we also find the optimal K-value needed for the KMeans clustering.
I have used Elbow method for find the optimal K-value for KMeans clustering.
```
K = range(2, 50, 2)
for k in K:
   k_means = KMeans(n_clusters=k, random_state=10).fit(pcaReducedDim)
   k_means.fit(pcaReducedDim)
   distortions.append(sum(np.min(cdist(pcaReducedDim, k_means.cluster_centers_, 'euclidean'), axis=1)) / vector.shape[0])
```
With the help of Elbow plot, we find that the optimal K-value is to be 10.

### Clustering():

In this function I am creating a KMeans model and preparing the cluster data.
This function takes in 2 argument which are PCA dimension reduced matrix and the actual list of data from the chooseFiles method.
```
model = KMeans(n_clusters=optCluster, init='k-means++', n_init=10, max_iter=250,verbose=0, random_state=None, precompute_distances='auto',copy_x=True, n_jobs=None, algorithm='auto')
```
We pass cluster size a parameter to the model, we are forming 10 cluster in our case.

We get the documents in each cluster with the help of model.labels_.
With the help of model.cluster_centers_ we iterate through each cluster, and in each cluster using model.labels_ we iterate through each document and append the data present in each document into a list.
The clusterData list is returned in this method.

### Summarization()
This function takes in 1 argument which is cluster data from the clustering method.
In this function I am doing extractive text summarization using genism.
In this method we iterate through cluster data and then convert the cluster data into a string and then apply genism summarize on each cluster data and select top 10 sentence in the cluster data.
```
summarized = summarize(strData, ratio=1, split=True)[0:10]
```
In this function we write top 10 sentence of each cluster data into file.
We write each cluster data into a separate cluster files.

 ------
## Assumptions/Bugs
----
- This code only handles english language.
- I could implement the text summarization for 1% of data files.The instance stop after sometime when I run for 10% data.

 ------
## How to Execute the Code
----
- We need to clone the github repository to the local instance using ``` git clone https://github.com/Rajashekar-Veer/cs5293sp20-project2.git ``` code.
- Navigate to ../cs5293sp-project2 directory.
- The dataset must be placed in the inputfiles folder before running the program.
- Run the below code to execute the project.
```
pipenv run python project1/summarization.py
```
- Executing the program with this command line argument will read all files in the inputfiles folder ending with .json. The program does the text summarization and clusters the data based on the optimal cluster size and save the top 10 sentence of the documents present in each cluster to outputFiles folder.



