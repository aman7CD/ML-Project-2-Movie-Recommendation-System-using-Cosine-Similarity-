
## Importing the dependencies
import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import difflib

from sklearn.metrics.pairwise import cosine_similarity



## Data Colection and Preprocessing
data = pd.read_csv("/kaggle/input/authentic-dataset-for-movie-recommendation-system/movies.csv")

data.isnull().sum()

data.head()

feature = data["genres"]+" "+data["keywords"]+" "+data["tagline"]+" "+data["director"]+" "+data["cast"]

feature.fillna("", inplace=True)



## Converting data from text to numerical
vt = TfidfVectorizer(lowercase=True)

extracted_features = vt.fit_transform(feature)



## Finding the Cosine Similarity
similarity = cosine_similarity(extracted_features)

movie_name = input("enter a movie name: ")


    
list_of_titles = data["title"].tolist()

find_close_match = difflib.get_close_matches(movie_name,list_of_titles)

close_match = find_close_match[0]

index_of_close_match = data[data["title"]==close_match].index[0]

similarity1 = list(enumerate(similarity[index_of_close_match]))

sorted_similarity1 = sorted(similarity1, reverse=True, key=lambda x:x[1])

c=1
for i in sorted_similarity1:
    a = i[0]
    b = data[data["index"]==a].title.values[0]
    c=c+1
    if c<25:
        print(b)
  
  
 
