import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval


x1=pd.read_csv(r"C:\Users\user\OneDrive\Desktop\data science\ML_rsume_projects\tmdb_5000_credits.csv")
x2=pd.read_csv(r"C:\Users\user\OneDrive\Desktop\data science\ML_rsume_projects\tmdb_5000_movies.csv")


#merge the two dataframes
x1.columns = ['id','tittle','cast','crew']
x2= x2.merge(x1,on='id')


tfidf = TfidfVectorizer(stop_words='english')
x2['overview'] = x2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(x2['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(x2.index, index=x2['title']).drop_duplicates()




def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    
   
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return x2['title'].iloc[movie_indices]





features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    x2[feature] = x2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

x2['director'] = x2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']

for feature in features:
    x2[feature] = x2[feature].apply(get_list)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    x2[feature] = x2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
x2['soup'] = x2.apply(create_soup, axis=1)


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(x2['soup'])


cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

x2 = x2.reset_index()
indices = pd.Series(x2.index, index=x2['title'])


#code driver
x=input('enter the movie name:')
X=x.title()
movies=get_recommendations(X, cosine_sim2)
print(movies)