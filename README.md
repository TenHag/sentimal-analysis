# sentiment-analysis
import pandas as pd

path = r'C:\Users\laptop\Desktop\ADITYA\MovieD\IMDB.csv'
df = pd.read_csv(path)
df.head()
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
docs = np.array(['I am Aditya, studying in XIE'
                 'Please like,share,comment and Subscribe to my channel'
                 'thanks for all the support to my channel'])

bag = vect.fit_transform(docs)
print(vect.vocabulary_)
print(bag.toarray())
