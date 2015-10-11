# collaborative_filtering

Collaborative filtering techniques implemented in Python's theano package. Thus far these include matrix factorization.

More information can be found in [this paper](http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf).

# requirements

* numpy
* pandas
* theano

To install these with `pip` run `pip install numpy, pandas, theano` in a shell.

# usage

### data

The `data.py` module aids in downloading of datasets to use.

### matrix factorization

The `matrix_factorization.py` module allows the user to map a set of users and a set of items into the same vector space. Then distances between entities in that space (either users or items) may be examined to find relationships between any user and item.

Running the `matrix_factorization.py` module will attempt to load the [MovieLens dataset](http://grouplens.org/datasets/movielens/latest/) (see `data.py` for easy download of this dataset).
