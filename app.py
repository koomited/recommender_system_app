import sys
import os
import pandas as pd
import numpy as np
import json

from  utils import AlternatingLeastSquare

API_KEY = "2c830e4a404d252f6488cfd8d593e9d4"

import streamlit as st
st.set_page_config(layout="wide")


# eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyYzgzMGU0YTQwNGQyNTJmNjQ4OGNmZDhkNTkzZTlkNCIsIm5iZiI6MTczMDQ4NDIyOC41MDkyNzA0LCJzdWIiOiI2NzI1MTRkNWU3YjM3YzczYzY5ZDMwNDUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.w47ESBaslD6AZE_fiVRMuYnltBQcAuLyON0Cwrufsb0

import requests

def fetch_poster(tmdb_id=None, imdb_id=None, api_key='YOUR_TMDB_API_KEY'):
    base_url = "https://api.themoviedb.org/3"
    
    # Use the appropriate endpoint based on the ID available
    if tmdb_id:
        url = f"{base_url}/movie/{tmdb_id}"
    elif imdb_id:
        url = f"{base_url}/find/{imdb_id}?external_source=imdb_id"
    else:
        raise ValueError("Either tmdb_id or imdb_id must be provided.")

    # Parameters for the request
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    data = response.json()
    
    # For IMDb ID, data is nested under 'movie_results'
    if imdb_id:
        if data['movie_results']:
            movie_data = data['movie_results'][0]
        else:
            raise ValueError("No movie found for the provided IMDb ID.")
    else:
        movie_data = data

    # Retrieve the poster path
    poster_path = movie_data.get("poster_path")
    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/original{poster_path}"
        return poster_url
    else:
       pass


# map_movie_to_idx = json.load( open( "data/map_movie_to_idx.json" ) )

with open("map_idx_to_movie", "r") as fp:
    map_idx_to_movie= json.load(fp)

def recommendation_for_new_user_stream(movies_dir,movie_title, movie_rating, lambd, tau):

    movies_df = pd.read_csv(movies_dir)

    movie_id = movies_df[movies_df["title"]==movie_title]["movieId"]
    movie_id = movie_id.item()

    get_this_movie_index = map_idx_to_movie.index(movie_id)

    # get the item_vector
    items_latents = pd.read_csv("items_latents.csv", index_col=0)
    items_latents = items_latents.to_numpy()
    item_vector = items_latents[:, get_this_movie_index]

    #get this item baias
    item_biases = pd.read_csv("item_biases.csv", index_col=0)
    item_biases = item_biases.to_numpy().ravel()
    item_bias = item_biases[get_this_movie_index]



    #compute new user embedding
    new_user_embdding = np.linalg.inv(lambd*np.outer(item_vector, item_vector)+tau*np.eye(10))@(lambd*item_vector*(movie_rating - item_bias))

    # compute the score 
    scores = (new_user_embdding@items_latents + 0.05*item_biases)
    

    movies_may_be_recommended_indexes = np.argsort(scores)[-20:][::-1]
    

    movies_to_recommend_ids = []
    for index in movies_may_be_recommended_indexes:
        
        movies_to_recommend_ids.append(map_idx_to_movie[index])

    links_data = pd.read_csv("links.csv")

    
    movie_timbds = []
    movies_names = []
    for movie_id in movies_to_recommend_ids:
        movies_names.append(model_object.get_movie_title_by_id(movies_dir, movie_id))
        movie_timbd_id = links_data[links_data["movieId"]== movie_id]["tmdbId"]
        movie_timbds.append(movie_timbd_id.item())

    
        
    return movies_names, movie_timbds



model_object = AlternatingLeastSquare("ratings.csv", 10)


# model_object.data_indexing()



def get_trendy_movies_names_posters():
    with open("sum_rating_per_movie", "r") as fp:
        sum_rating_per_movie= json.load(fp)

    links_data = pd.read_csv("links.csv")
    
    trendy_movies_index = np.argsort(np.array(sum_rating_per_movie))[-20:][::-1]
    movies_names = []
    movies_posters = []
    for index in trendy_movies_index:
        try:
            movie_id = map_idx_to_movie[index]

            movie_timbd_id = links_data[links_data["movieId"]== movie_id]["tmdbId"]
            movie_timbd_id = movie_timbd_id.item()
            movies_names.append(model_object.get_movie_title_by_id("movies.csv", movie_id))
            movies_posters.append(fetch_poster(tmdb_id=movie_timbd_id, api_key=API_KEY))
        except:
            continue
    return  movies_names , movies_posters

movies_names, movies_posters = get_trendy_movies_names_posters()


st.write(f"""
             <div class="title", style="text-align: center">
             <span style="font-size:32px;"> MovieLens Recommender System 👋</span>
             </div>
             """, unsafe_allow_html=True)
st.markdown('<style> div.block-container{padding-top:3rem;}</style>', unsafe_allow_html=True)
    

st.sidebar.title("Menu")

# st.sidebar.button("Menu")
# st.sidebar.button("Preferences")
# st.sidebar.button("Search")
# st.sidebar.button("Recommendation")

page = st.sidebar.radio("", ["Popular",  "Recommendations", "Search"])

if page=='Popular':
    st.write("##")
    st.subheader("Trending Now")
    # for movie_name, movie_poster in zip(movies_names, movies_posters):
    #     st.image(movie_poster, caption=movie_name, width=150)
    num_cols = 5
    for i in range(0, len(movies_posters), num_cols): 
        cols = st.columns(num_cols) 
        
        for j in range(num_cols):
            if i + j < len(movies_posters):  # Check if the index is within range
                cols[j].image(movies_posters[i + j], caption=movies_names[i + j], use_column_width=True)

elif page=="Search":
    # pass
    st.subheader("Search for the movie you are looking for!!!!🧐🧐")
    st.text_input("",placeholder="Insert your key words here.")
elif page=="Recommendations":
    st.session_state.sidebar_state = "collapsed" #if st.session_state.sidebar_state == "expanded" else "expanded"
    
    # pass
    movies_df = pd.read_csv("movies.csv")
    movie_titles = movies_df["title"].tolist()
    selected_movie = st.selectbox("The first movies you are rating", movie_titles)
    # movie_id = st.("Movie you rate")
    # st.write(movie_id)
    movie_rate = st.number_input("The rate you give", min_value=0, max_value=5)

    st.markdown("""
    <style>
    div.stSpinner > div {
    text-align:center;
    align-items: center;
    justify-content: center;
    }
    </style>""", unsafe_allow_html=True)

    spinner = st.spinner('Wait for it...',)





    if st.button("What you may also like!", use_container_width=True, type="primary"):
        with spinner:

            movies_names_recom, movie_timbds = recommendation_for_new_user_stream("movies.csv",selected_movie, movie_rate, 0.1, 0.2)
            movies_posters_recomm = []
            for  movie_timbd in movie_timbds :
                movies_posters_recomm.append(fetch_poster(tmdb_id=movie_timbd, api_key=API_KEY))
            num_cols = 5
            for i in range(0, len(movies_posters_recomm), num_cols): 
                cols = st.columns(num_cols) 
                
                for j in range(num_cols):
                    if i + j < len(movies_posters_recomm):  # Check if the index is within range
                        try:
                            cols[j].image(movies_posters_recomm[i + j], caption=movies_names_recom[i + j], use_column_width=True)
                        except:
                            continue


