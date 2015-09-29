import os
import pickle

import theano
from theano import tensor as T
import pandas as pd
import numpy as np


def load_var_features_dfs(fp='var_features_dfs.pkl'):

    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            res = pickle.load(f)
    else:
        res = None

    return res

def save_var_features_dfs(dfs, fp='var_features_dfs.pkl'):

    with open(fp, 'wb') as f:
        pickle.dump(dfs, f)

def get_latent_feature_dfs(ratings=None, fp='user_item_features_dfs.pkl', n_latent_features=20):
    
    res = load_var_features_dfs(fp)
    if res:
        return res

    user_index = sorted(ratings.index.levels[0])
    item_index = sorted(ratings.index.levels[1])

    shape_user = (n_latent_features, len(user_index))
    shape_item = (n_latent_features, len(item_index))

    scale_user = np.sqrt(2. / (sum(shape_user)))
    scale_item = np.sqrt(2. / (sum(shape_item)))

    user_vals = np.random.randn(*shape_user) * scale_user
    item_vals = np.random.randn(*shape_item) * scale_item

    # Add biases and weights to users/items
    bias_weight = np.array([1, 0])[:, None]

    user_vals = np.concatenate((np.repeat(bias_weight, user_vals.shape[1], axis=1), user_vals))
    item_vals = np.concatenate((np.repeat(bias_weight[::-1], item_vals.shape[1], axis=1), item_vals))

    users = pd.DataFrame(user_vals, columns=user_index)
    items = pd.DataFrame(item_vals, columns=item_index)

    return users, items

def build_model(var1_name='var1', var2_name='var2'):

    ratings = T.vector('ratings')
    var1_matrix = T.matrix('{}_matrix'.format(var1_name))
    var2_vector = T.vector('{}_vector'.format(var2_name))

    predictions = T.dot(var2_vector, var1_matrix)

    # TODO: figure out merits of sum vs mean
    prediction_error = ((ratings - predictions) ** 2).sum()
    l2_penalty = (var2_vector ** 2).sum() + (var1_matrix ** 2).sum().sum()

    cost = prediction_error + l2_penalty

    var1_grad = T.grad(cost, var1_matrix, consider_constant=[var2_vector])
    var2_grad = T.grad(cost, var2_vector, consider_constant=[var1_matrix])

    f = theano.function(inputs=[ratings, var1_matrix, var2_vector], outputs=[cost, var1_grad, var2_grad])

    return f

def train(data, users_all, items_all, level=0, max_epochs=100):

    epoch = 0

    try:
        while epoch < max_epochs:

            epoch += 1

            for (item_idx, ratings_series) in data.groupby(level=level):

                user_idxs = ratings_series.index.get_level_values(0)

                users = users_all[user_idxs]
                item = items_all[item_idx]
                c, ug, ig = f(ratings_series, users, item)

                users_all[user_idxs] = users - lr * ug
                items_all[item_idx] = item - lr * ig
    except KeyboardInterrupt:
        print('stopping training early on epoch {}'.format(epoch))
    finally:
        save_var_features_dfs((users_all, items_all), fp=var_features_fp)


if __name__ == '__main__':

    # TODO: adjust adadelta so that it can be used on a subset of the space
    # TODO: Add biases

    movies_ratings_fp = 'ml-latest/ratings.csv'  # fp to movie ratings csv
    var_features_fp = 'user_movie_features_dfs.pkl'  # fp to pickled variable_feature dataframes

    n_latent_features = 20
    lr = 0.05

    print('loading data')
    s = pd.read_csv(movies_ratings_fp, index_col=['userId', 'movieId'])['rating']

    users, movies = get_latent_feature_dfs(s, fp=var_features_fp, n_latent_features=n_latent_features)

    print('building model')
    f = build_model()

    print('training')
    train(s, users, movies, level=1)
