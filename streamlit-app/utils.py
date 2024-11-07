import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.stats import norm
from collections import defaultdict
import random
from tqdm import tqdm 
import pandas as pd

import matplotlib.pylab as pylab

np.random.seed(42)

params = {'pdf.fonttype': 3, 'axes.labelsize': 18, 'xtick.labelsize':18
, 'ytick.labelsize':18, 'legend.fontsize':18, "font.size":18}
plt.rcParams.update(params)

class AlternatingLeastSquare:
    def __init__(self, data_dir, embedding_dim:int) -> None:
        self.data_dir = data_dir
        self.factors_number = embedding_dim
        # Mappings for user and movie indexes
        self.map_user_to_idx = {}
        self.map_idx_to_user = []
        self.data_by_user_id = defaultdict(list)

      

        self.map_movie_to_idx = {}
        self.map_idx_to_movie = []
        self.data_by_movie_id = defaultdict(list)

        
       
        

    def data_indexing(self):
        with open(self.data_dir, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])

                # Handle user indexing
                if user_id not in self.map_user_to_idx:
                    self.map_user_to_idx[user_id] = len(self.map_idx_to_user)
                    self.map_idx_to_user.append(user_id)
                user_idx = self.map_user_to_idx[user_id]

                # Handle movie indexing
                if movie_id not in self.map_movie_to_idx:
                    self.map_movie_to_idx[movie_id] = len(self.map_idx_to_movie)
                    self.map_idx_to_movie.append(movie_id)
                movie_idx = self.map_movie_to_idx[movie_id]

                # Append data for user and movie
                self.data_by_user_id[user_idx].append((movie_id, rating))
                self.data_by_movie_id[movie_idx].append((user_id, rating))
        self.data_by_movie_id = [data for data in self.data_by_movie_id.values()]
        self.data_by_user_id = [data for data in self.data_by_user_id.values()]

        



    def get_data_by_user_id(self, user_id):
        position = self.map_idx_to_user.index(user_id)
        user_data = self.data_by_user_id[position]
        return user_data

    def get_data_by_movie_id(self, movie_id):
        position = self.map_idx_to_movie.index(movie_id)
        movie_data = self.data_by_movie_id[position]
        return movie_data

    def average_rating_per_movie(self):
        average_rating_per_movie = []
        for movie in self.data_by_movie_id:
            rating_sum = 0
            for rating in movie:
                rating_sum+=rating[1]
            average_rating_per_movie.append(rating_sum/len(movie))
        return average_rating_per_movie

    def sum_rating_per_movie(self):
        sum_rating_per_movie = []
        for movie in self.data_by_movie_id:
            rating_sum = 0
            for rating in movie:
                rating_sum+=rating[1]
            sum_rating_per_movie.append(rating_sum)
        return sum_rating_per_movie

    def plot_power_law(self, fig_name):
        number_of_movies_per_user = [len(user_movies) for user_movies in self.data_by_user_id]
        number_of_users_per_movie = [len(movie_users) for movie_users in self.data_by_movie_id]
        minimum_number_rating = min(number_of_movies_per_user)

        users_ratings_number_frequency = {num_ratings:  number_of_movies_per_user.count(num_ratings)  for num_ratings in number_of_movies_per_user }
        movies_ratings_number_frequency = {num_ratings: number_of_users_per_movie.count(num_ratings)  for num_ratings in number_of_users_per_movie }

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(users_ratings_number_frequency.keys(),  users_ratings_number_frequency.values(),
                  marker = ".", ls = "none", color = "m", label="Users")
        ax.loglog(movies_ratings_number_frequency.keys(), movies_ratings_number_frequency.values() ,
                  marker = ".", ls = "none", color = "blue", label="Movies")
        ax.axvline(x = minimum_number_rating, color = 'r', ls="--")
        ax.text(10**1*2.3, 10**3*1.8, "Minimun ratings number", color ="r")#fontsize = 12
        ax.set_xlabel("Degree")
        ax.set_ylabel("Frequencies")
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi = 1000)
        
        plt.show()


    def plot_average_rating_hist(self, fig_name):
   
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(self.average_rating_per_movie(),stat="probability", bins=10, kde=True, kde_kws={"bw_adjust":3}, color="m")
        ax.set_xlabel("Average ratings")
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def line_plot(self, data_train, data_test, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(data_train)+1), data_train,  color='blue', lw=1, label='Training')
        ax.plot(range(1,len(data_test)+1), data_test, color='red', lw=1, label='Testing')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.legend(bbox_to_anchor=(1.06, .6), loc="center left", frameon=False)

        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def plot_training_loss_only(self, losses_train, xaxis, yaxis, fig_name):

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1,len(losses_train)+1), losses_train,  color='blue', lw=1, label='Training')
        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)

        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=1000)

        plt.show()

    def alternating_least_square(self, data_user, data_movie, lambd=0.05,tau=0.01, gamma=0.05, epochs = 1000):
        # number of users
        M = len(data_user)
        # number of items
        N = len(data_movie)
        user_biases = np.zeros(M)
        item_biases = np.zeros(N)
        
        
        self.users_latents = np.random.normal(0, 1/np.sqrt(self.factors_number),(M, self.factors_number))
        self.items_latents = np.random.normal(0,1/np.sqrt(self.factors_number),(self.factors_number, N))

        losses_train = []
        rmses_train = []
        losses_test=[]
        rmses_test = []

        with tqdm(range(epochs), unit='epoch') as tepochs:
            tepochs.set_description('Training')
            for i in tepochs:
                # users biases computation
                for m in range(M):
                    bias = 0
                    item_counter = 0
                    for n, r in data_user[m]:
                        user_embedding = self.users_latents[m,:]
                        item_embedding  = self.items_latents[:,self.map_movie_to_idx[n]]
                        bias += lambd*(r - item_biases[self.map_movie_to_idx[n]] -np.dot(user_embedding, item_embedding)) #
                        item_counter += 1
                    bias = bias/(lambd * item_counter + gamma) 
                    user_biases[m] = bias
                    # user embedding
                    # fisrt term of user embedding computation
                    f_user_embedding  = np.zeros((self.factors_number, self.factors_number))

                    # #second term of user embedding computation
                    s_user_embedding  = np.zeros((self.factors_number, 1))
                    for n, r in data_user[m]:
                        item_embedding = self.items_latents[:, self.map_movie_to_idx[n]]
                        f_user_embedding += np.outer(item_embedding, item_embedding) 
                        s_user_embedding+= (item_embedding*(r-user_biases[m]-item_biases[self.map_movie_to_idx[n]])).reshape(self.factors_number,1)
                    self.users_latents[m, :] = (np.linalg.inv(lambd*f_user_embedding + tau*np.identity(self.factors_number))@(lambd*s_user_embedding )).reshape(1,self.factors_number)
                    
                # items biases and embedding computation
                for n in range(N):
                    bias = 0
                    user_counter = 0
                    for m, r in data_movie[n]:
                        user_embedding = self.users_latents[self.map_user_to_idx[m],:]
                        item_embedding = self.items_latents[:,n]
                        bias += lambd*(r - user_biases[self.map_user_to_idx[m]]-np.dot(user_embedding, item_embedding)) #
                        user_counter += 1
                    bias = bias/(lambd*user_counter + gamma) 
                    item_biases[n] = bias

                    # items embedding
                    # fisrt part of item embedding
                    f_item_embedding  = np.zeros((self.factors_number, self.factors_number))
                    # second part of item embedding
                    s_item_embedding  = np.zeros((1, self.factors_number))
                    for m, r in data_movie[n]:
                        user_embedding = self.users_latents[self.map_user_to_idx[m], :]
                        f_item_embedding += np.outer(user_embedding, user_embedding) 
                        s_item_embedding+= (user_embedding*(r-item_biases[n]-user_biases[self.map_user_to_idx[m]])).reshape(1, self.factors_number)
                    item_embedding = (np.linalg.inv(lambd*f_item_embedding + tau*np.identity(self.factors_number))@(lambd*s_item_embedding.reshape(self.factors_number,1)))
                    item_embedding = item_embedding.reshape(1, self.factors_number)
                    self.items_latents[:, n] = item_embedding
                    

                loss_train, rmse_train= self.loss_rmse_function(data_user, user_biases, item_biases, lambd = lambd, tau=tau, gamma = gamma)
                loss_test, rmse_test = self.loss_rmse_function(self.data_by_user_test, user_biases, item_biases, lambd = lambd, tau=tau, gamma = gamma)
                losses_train.append(loss_train)
                rmses_train.append(rmse_train)
                losses_test.append(loss_test)
                rmses_test.append(rmse_test)
                tepochs.set_postfix(test_rmse=rmse_test)
 
        self.items_biases = item_biases      
        self.users_biases = user_biases      
        return user_biases, item_biases, losses_train, rmses_train, losses_test, rmses_test

        
    def  loss_rmse_function(self, data, user_biases, item_biases, lambd = 0.5, tau=0.01, gamma = 0.5):
        M = len(data)
        # initialize loss for iterations
        loss= gamma*np.sum(user_biases**2)/2 + gamma*np.sum(item_biases**2)/2
        item_vector_loss =np.sum(self.items_latents**2)*tau/2
        
        rmse_list=[]
        user_vector_loss = 0
        # for each user
        for m in range(M):
            ratings_loss = []
            
            user_embedding = self.users_latents[m,:]
            user_vector_loss += np.dot(user_embedding, user_embedding)            
            # for each item rated by the user
            for n, r in data[m]:
                item_embedding = self.items_latents[:,self.map_movie_to_idx[n]]
                ratings_loss.append((r-user_biases[m] -item_biases[self.map_movie_to_idx[n]]-np.dot(user_embedding, item_embedding ))**2)# 
                rmse_list.append((r-user_biases[m] -item_biases[self.map_movie_to_idx[n]] -np.dot(user_embedding, item_embedding ) )**2)# 
            loss+= lambd*sum(ratings_loss)/2 
            
        loss += tau*user_vector_loss/2 + item_vector_loss
        rmse = np.sqrt(np.mean(rmse_list))
        return loss, rmse

    def compute_items_scores(self):
        scores_for_items = self.users_latents@self.items_latents + 0.05*self.items_biases.reshape(1,-1)
        self.items_scores = scores_for_items
        # return scores_for_items 

    def get_movie_title_by_id(self, movies_dir, movie_id):
        movies_df = pd.read_csv(movies_dir)
        movies_df.index= movies_df["movieId"]
        movies_df.drop(columns="movieId")
        movie_title = movies_df.loc[movie_id,"title"]
        return movie_title

    def recommendation_for_user(self, movies_dir, user_id):
        self.compute_items_scores()
        user_index = self.map_user_to_idx[user_id]
        items_scores = self.items_scores[user_index,:]
        movies_may_be_recommended_indexes = np.argpartition(items_scores, -25)[-25:]

        # check if the user already rated one of them
        for movie_id, _ in self.data_by_user_id[user_index]:
            movie_index = self.map_movie_to_idx[movie_id]
            if movie_index in movies_may_be_recommended_indexes:
                movies_may_be_recommended_indexes = np.setdiff1d(movies_may_be_recommended_indexes, [movie_index])

        if len( movies_may_be_recommended_indexes)>5:
            movies_may_be_recommended_indexes = movies_may_be_recommended_indexes[:5]
            
        movies_to_recommend_ids = []
        for index in movies_may_be_recommended_indexes:
            movies_to_recommend_ids.append(self.map_idx_to_movie[index])
        
        movies_names = []
        for movie_id in movies_to_recommend_ids:
            movies_names.append(self.get_movie_title_by_id(movies_dir, movie_id))
            
        print(f"User {user_id} may also like:{movies_names}")

    def create_dummy_user(self, movie_id, rating):
         self.data_by_user_id.append([(movie_id, rating)])

    
    
    
    def recommendation_for_new_user(self, movies_dir, lambd, tau):

        get_new_user_movie_id = self.data_by_user_id[-1][0][0]

        rating = self.data_by_user_id[-1][0][1]

        get_this_movie_index = self.map_movie_to_idx[get_new_user_movie_id]

        # get the item_vector
        item_vector = self.items_latents[:, get_this_movie_index]

        #get this item baias
        item_bias = self.items_biases[get_this_movie_index]



        #compute new user embedding
        new_user_embdding = np.linalg.inv(lambd*np.outer(item_vector, item_vector)+tau*np.eye(self.factors_number))@(lambd*item_vector*(rating - item_bias))

        # compute the score 
        scores = (new_user_embdding@self.items_latents + 0.05*self.items_biases.reshape(1,-1))[0]
        

        movies_may_be_recommended_indexes = np.argsort(scores)[-11:]
        

        movies_to_recommend_ids = []
        for index in movies_may_be_recommended_indexes:
            
            movies_to_recommend_ids.append(self.map_idx_to_movie[index])
        
        movies_names = []
        for movie_id in movies_to_recommend_ids:
            movies_names.append(self.get_movie_title_by_id(movies_dir, movie_id))
        movies_names.reverse()
        
        return scores[movies_may_be_recommended_indexes], movies_names 

    def recommendation_for_new_user_stream(self, movies_dir,movie_title, movie_rating, lambd, tau):

        movies_df = pd.read_csv(movies_dir)

        movie_id = movies_df[movies_df["title"]==movie_title]["movieId"]
        movie_id = movie_id.item()

        get_this_movie_index = self.map_movie_to_idx[movie_id]

        # get the item_vector
        items_latents = pd.read_csv("data/items_latents.csv", index_col=0)
        items_latents = items_latents.to_numpy()
        item_vector = items_latents[:, get_this_movie_index]

        #get this item baias
        item_biases = pd.read_csv("data/item_biases.csv", index_col=0)
        item_biases = item_biases.to_numpy().ravel()
        item_bias = item_biases[get_this_movie_index]



        #compute new user embedding
        new_user_embdding = np.linalg.inv(lambd*np.outer(item_vector, item_vector)+tau*np.eye(self.factors_number))@(lambd*item_vector*(movie_rating - item_bias))

        # compute the score 
        scores = (new_user_embdding@items_latents + 0.05*item_biases)
        

        movies_may_be_recommended_indexes = np.argsort(scores)[-20:][::-1]
        

        movies_to_recommend_ids = []
        for index in movies_may_be_recommended_indexes:
            
            movies_to_recommend_ids.append(self.map_idx_to_movie[index])

        links_data = pd.read_csv("../data/ml-25m/links.csv")

        
        movie_timbds = []
        movies_names = []
        for movie_id in movies_to_recommend_ids:
            movies_names.append(self.get_movie_title_by_id(movies_dir, movie_id))
            movie_timbd_id = links_data[links_data["movieId"]== movie_id]["tmdbId"]
            movie_timbds.append(movie_timbd_id.item())

        
            
        return movies_names, movie_timbds



    def train_test_split(self):
        self.map_user_to_idx = {}
        self.map_idx_to_user = []
        self.data_by_user_train = []
        self.data_by_user_test = []

        self.map_movie_to_idx = {}
        self.map_idx_to_movie = []
        self.data_by_movie_train = []
        self.data_by_movie_test = []

        with open(self.data_dir, "r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])

                # Assign randomly to train or test
                flip_coin = random.random()
                is_train = flip_coin < 0.8

                # Handle user mapping
                if user_id not in self.map_user_to_idx:
                    self.map_idx_to_user.append(user_id)
                    self.map_user_to_idx[user_id] = len(self.map_idx_to_user) - 1
                    self.data_by_user_train.append([])
                    self.data_by_user_test.append([])

                user_idx = self.map_user_to_idx[user_id]

                # Handle movie mapping
                if movie_id not in self.map_movie_to_idx:
                    self.map_idx_to_movie.append(movie_id)
                    self.map_movie_to_idx[movie_id] = len(self.map_idx_to_movie) - 1
                    self.data_by_movie_train.append([])
                    self.data_by_movie_test.append([])

                movie_idx = self.map_movie_to_idx[movie_id]

                # Add to train or test set for both user and movie
                if is_train:
                    self.data_by_user_train[user_idx].append((movie_id, rating))
                    self.data_by_movie_train[movie_idx].append((user_id, rating))
                else:
                    self.data_by_user_test[user_idx].append((movie_id, rating))
                    self.data_by_movie_test[movie_idx].append((user_id, rating))

    def plot_embedding(self, movies_dir, points_number, fig_name):
        items_x = self.items_latents[0,:][:points_number]
        items_y = self.items_latents[1,:][:points_number]
        
       
        
        movies_ids = []
        for index in range(len(items_x)):
            
            movies_ids.append(self.map_idx_to_movie[index])
            
      
        
        movies_names = []
        for movie_id in movies_ids:
            movies_names.append(self.get_movie_title_by_id(movies_dir, movie_id))
        
        fig, ax = plt.subplots()
        ax.scatter(items_x, items_y, color="m")
        ax.set_xlabel("Embedding dimenssion 1")
        ax.set_ylabel("Embedding dimenssion 2")
        

        for i, txt in enumerate(movies_names):
            if i%2:
                continue
            ax.annotate(txt[:10], (items_x[i], items_y[i]))
        plt.savefig(f"plots/{fig_name}.pdf", format="pdf", bbox_inches="tight", dpi=2000)
        
            
       








