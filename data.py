from urllib.request import urlretrieve
import os

dataset_dict = {
    'movielens': ('http://files.grouplens.org/datasets/movielens/ml-latest.zip', 'zip', 'ml-latest')
}

def get_data(dataset):

    if dataset not in dataset_dict:
        print('No url for {} dataset'.format(dataset))
        return

    url, file_type, write_name = dataset_dict[dataset]

    if os.path.isfile(write_name) or os.path.isdir(write_name):
        print('Dataset already downloaded to {}'.format(write_name))
        return

    local_filename, headers = urlretrieve(url)

    if file_type == 'zip':
        import zipfile
        f = zipfile.ZipFile(local_filename)
        f.extractall()

    return write_name

if __name__ == '__main__':

    get_data('movielens')
