# Classification of Document using Graph Model

## Introduction
Graph-based methods offer a versatile approach to document classification by representing documents and their relationships as graphs. 
- Document Representation as Graphs
- Graph Construction
- Graph Similarity
- Classification Algorithms

## Methodology
- ### Web Scraping
  The technique we use in for web scraping is often referred to as “basic web scraping”. It involve HTTP request to web pages, retrieving their HTML content, parsing the HTML using a library like BeautifulSoup, and extracting desired information from the pared HTML.
  
  ![image](https://github.com/Zainulaben/Classification-of-Document-using-Graph-Model/assets/171002327/8c1559eb-a84b-4f45-a973-628265eec108)

- ### Data Cleaning
  In data cleaning process first we remove all the HTML tags, then remove stopping words, then perform lemmatization the purpose of lemmatization is that to reduce the word to their base form and also remove punctuation. This process will be perform after web scrapping.

  ![image](https://github.com/Zainulaben/Classification-of-Document-using-Graph-Model/assets/171002327/f1700844-68ac-448a-97b3-7b1b63a178af)

- ### Graph Creation
  In this we make edges between words. All the word act like nodes and the sequence from one word to other word make edge. “Example This is a sample text for building a graph. Graph visualization is an important aspect of data analysis”.

  ![image](https://github.com/Zainulaben/Classification-of-Document-using-Graph-Model/assets/171002327/0d879cae-ac54-47b6-859f-a3767e8371c9)

- ## Data Classification
  After implementation of MCS, we have stored all the maximum distances of testing graphs with all training graphs.
  Then we have sort all the distances for each testing graph and find k nearest distances. We have used k value equal to 3. Then we have assign labels to each testing graph based on the nearest distances graphs’ labels. In this we use **K Nearest Neighbors(KNN)** for classification of document.

  ![image](https://github.com/Zainulaben/Classification-of-Document-using-Graph-Model/assets/171002327/26def339-137c-4c67-8761-1965a91d57cc)

- ## Reults
  We have plotted confusion matrix which shows the true labels and predicted labels. All the predicted labels are predicted correctly as it can be seen.
  So, it is also showing that accuracy is 100% for our test data.

  ![image](https://github.com/Zainulaben/Classification-of-Document-using-Graph-Model/assets/171002327/4db63809-70ff-4c3c-ac8f-f5d66ccc4aeb)

