import os
import pickle

import theano
from theano import tensor as T
import pandas as pd
import numpy as np
from sklearn import cross_validation


def load_var_features_dfs(fp):
    """
    Load DataFrames that describe the variables in the shared latent feature space
    :param fp: filepath to load from
    :return: two DataFrames or None if DataFrames could not be found/loaded
    """

    if os.path.isfile(fp):
        with open(fp, 'rb') as f:
            res = pickle.load(f)
    else:
        res = None

    return res

def save_var_features_dfs(dfs, fp):
    """
    Save DataFrames that describe the variables in the shared latent feature space
    :param dfs: length-2 iterable of DataFrames
    :param fp: filepath to save to
    """

    with open(fp, 'wb') as f:
        pickle.dump(dfs, f)

def get_latent_feature_dfs(ratings=None, fp='user_item_features_dfs.pkl', n_latent_features=20):
    """
    Get DataFrames that describe the variables in the shared latent feature space
    :param ratings: Series of ratings that has a multiindex w/ levels (user, item)
    :param fp: filepath to load DataFrames from
    :param n_latent_features: dimensionality of latent feature space
    :return: tuple of two DataFrames respectively describing users and items in the latent feature space
    """
    
    res = load_var_features_dfs(fp)
    if res:
        return res

    # Do Glorot initialization for features
    user_index = sorted(ratings.index.levels[0])
    item_index = sorted(ratings.index.levels[1])

    shape_user = (n_latent_features, len(user_index))
    shape_item = (n_latent_features, len(item_index))

    scale_user = np.sqrt(2. / (sum(shape_user)))
    scale_item = np.sqrt(2. / (sum(shape_item)))

    user_vals = np.random.randn(*shape_user) * scale_user
    item_vals = np.random.randn(*shape_item) * scale_item

    # Add biases and weights to users/items - user weights will be multiplied by item bias terms of 1 (and vice versa)
    bias_weight = np.array([1, 0])[:, None]

    user_vals = np.concatenate((np.repeat(bias_weight, user_vals.shape[1], axis=1), user_vals))
    item_vals = np.concatenate((np.repeat(bias_weight[::-1], item_vals.shape[1], axis=1), item_vals))

    # Return DataFrames holding the features
    users = pd.DataFrame(user_vals, columns=user_index, dtype=np.float32)
    items = pd.DataFrame(item_vals, columns=item_index, dtype=np.float32)

    return users, items

def build_model(reg_constant=0.1, var1_name='var1', var2_name='var2'):
    """
    Build MF model in theano
    :param reg_constant: regularization constant
    :param var1_name: name of first variable (e.g. users)
    :param var2_name: name of second variable (e.g. items)
    :return: theano function implementing MF model
    """

    ratings = T.vector('ratings')
    var1_vector = T.vector('{}_vector'.format(var1_name))
    var2_matrix = T.matrix('{}_matrix'.format(var2_name))

    predictions = T.dot(var1_vector, var2_matrix)

    prediction_error = ((ratings - predictions) ** 2).sum()
    l2_penalty = (var1_vector ** 2).sum() + (var2_matrix ** 2).sum().sum()

    cost = prediction_error + reg_constant * l2_penalty

    var1_grad = T.grad(cost, var1_vector)
    var2_grad = T.grad(cost, var2_matrix)

    f = theano.function(inputs=[ratings, var1_vector, var2_matrix], outputs=[cost, var1_grad, var2_grad])

    return f


def train(f, data, users_all, items_all, learning_rate, level):
    """
    Train the MF model
    :param f: theano model
    :param data: Series of ratings that has a multiindex w/ levels (user, item)
    :param users_all: DataFrame of all users in latent feature space
    :param items_all: DataFrame of all items in latent feature space
    :param learning_rate: learning rate used for SGD
    :param level: level of data's multiindex to groupby on for training
    """

    for (var1_idx, ratings_series) in data.groupby(level=level):

        if level == 0:
            var1_all, var2_all = users_all, items_all
            var2_idxs = ratings_series.index.get_level_values(1)
        elif level == 1:
            var1_all, var2_all = items_all, users_all
            var2_idxs = ratings_series.index.get_level_values(0)

        var1 = var1_all[var1_idx]
        var2 = var2_all[var2_idxs]

        c, g1, g2 = f(ratings_series, var1, var2)

        # TODO: implement momentum
        var1_all[var1_idx] -= learning_rate * g1
        var2_all[var2_idxs] -= learning_rate * g2


def validate(data, users_all, items_all):
    """
    Validate model
    :param data: Series of ratings that has a multiindex w/ levels (user, item)
    :param users_all: DataFrame of all users in latent feature space
    :param items_all: DataFrame of all items in latent feature space
    :return: error averaged over validation data
    """

    errors = []

    for (user_idx, user_ratings) in data.groupby(level=0):

        item_idxs = user_ratings.index.get_level_values(1).values

        user = users_all[user_idx]
        items = items_all[item_idxs]

        user_predictions = np.dot(user, items)

        user_errors = (user_ratings - user_predictions).abs()

        errors.extend(user_errors)

    error = np.mean(errors)

    return error


def main(f, data, users_all, items_all,
         learning_rate=0.000005, level=0, max_epochs=100,
         min_ratings_user=0, min_ratings_item=0,
         valid_frequency=0, perc_valid=0.1,
         save_frequency=0, save_fp=None):
    """
    Main loop for matrix factorization
    :param f: theano model
    :param data: Series of ratings that has a multiindex w/ levels (user, item)
    :param users_all: DataFrame of all users in latent feature space
    :param items_all: DataFrame of all items in latent feature space
    :param learning_rate: learning rate used for SGD
    :param level: level of data's multiindex to groupby on for training
    :param max_epochs: number of epochs before
    :param min_ratings_user: minimum number of ratings a user must have made
    :param min_ratings_item: minimum number of ratings an item must have
    :param valid_frequency: validation frequency (number epochs)
    :param perc_valid: percentage of eligible datapoints to validate on
    :param save_frequency: save frequency (number epochs) - if this is >0, must specify save_fp
    :param save_fp: filepath to save user and item feature matrices to
    """

    if min_ratings_user:
        data = data.groupby(level=0).filter(lambda s: len(s) >= min_ratings_user)

    if min_ratings_item:
        data = data.groupby(level=1).filter(lambda s: len(s) >= min_ratings_item)

    epoch = 0

    data_train, data_valid = cross_validation.train_test_split(data, test_size=perc_valid)

    try:
        while epoch < max_epochs:

            epoch += 1

            train(f, data_train, users_all, items_all, learning_rate, level)

            if valid_frequency and (epoch % valid_frequency == 0):

                error = validate(data, users_all, items_all)
                print('Validation error epoch {}: {}'.format(epoch, error))

            if save_frequency and (epoch % save_frequency == 0):

                save_var_features_dfs((users_all, items_all), fp=save_fp)

    except KeyboardInterrupt:
        print('Stopping training early on epoch {}'.format(epoch))
    finally:
        save_var_features_dfs((users_all, items_all), fp=save_fp)


if __name__ == '__main__':

    movies_ratings_fp = 'ml-latest/ratings.csv'  # fp to movie ratings csv
    var_features_fp = 'user_movie_features_dfs.pkl'  # fp to pickled variable_feature dataframes

    n_latent_features = 20

    print('loading data')
    data = pd.read_csv(movies_ratings_fp, index_col=['userId', 'movieId'])['rating']

    users, movies = get_latent_feature_dfs(data, fp=var_features_fp, n_latent_features=n_latent_features)

    print('building model')
    f = build_model()

    print('training')
    main(f, data, users, movies,
         level=1, min_ratings_item=10000, valid_frequency=5,
         save_frequency=10, save_fp=var_features_fp)
