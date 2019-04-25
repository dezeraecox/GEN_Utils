
"""This module offers a set of functions to dump a python dictionary indexed
by text strings to following file formats: `HDF5, INI, JSON`
"""

import os, pathlib
import pandas as pd
import numpy as np
from collections import Iterable
import matplotlib.pyplot as plt
from loguru import logger
logger.info('Import OK')

def h5store(filename, key, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put(key, df)
    store.get_storer(key).attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

def flatdict_to_h5(filename, dictionary, **kwargs):
    """
    Write single-nested dictionary to hdf5 file with metadata.

    Example usage:
    a = pd.DataFrame(data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))
    filename = '/tmp/data.h5'
    metadata = dict(local_tz='US/Eastern')
    h5store(filename, a, **metadata)
    with pd.HDFStore(filename) as store:
        data, metadata = h5load(store)
    """

    store = pd.HDFStore(filename)
    for key, df in dictionary.items():
        store.put(key, df)
        store.get_storer(key).attrs.metadata = kwargs
    store.close()


def flatten(coll):
    for i in coll:
            if isinstance(i, Iterable) and not isinstance(i, str):
                for subc in flatten(i):
                    yield subc
            else:
                yield i

def hdf_to_dict(raw_data, path=None):
    """Read a HDF5 file and return a nested dictionary with the complete file structure and all data.

    Example of usage::

    input_path = 'path.h5'
    raw_data = pd.HDFStore(input_path)
    test_dict = hdf_to_dict(raw_data, path='/')
    keys = test_dict.keys()
    for key in keys:
        test_dict[key]
    """

    logger.info(f'Running function with {path}')
    if not path:
        starting_keys = list(flatten(list(raw_data.walk())[0]))
    else:
        starting_keys = list(flatten(list(raw_data.walk(path))[0]))
    root = starting_keys.pop(0)
    logger.info(f'Starting keys: {starting_keys}')
    ddict = {}
    for key in starting_keys:
        try:
            logger.info(f'entered try loop with {key}')
            # new_sub_list = finder(raw_data, f'/{path}')
            # logger.info(f'Group found containing {new_sub_list}')
            ddict[key] = hdf_to_dict(raw_data, f'{root}/{key}')
            logger.info(f'Try loop failed with {key}')

        except:
            logger.info(f'Collecting df from {key}')
            ddict[key] = raw_data.get(f'{root}/{key}')
            logger.info(f'Successfully loaded df from {key}')

    return ddict


def dict_to_hdf(test, output_path, h5_group=''):
    """Write a nested dictionary to a HDF5 file, using keys as group names.

    If a dictionary value is a sub-dictionary, a group is created. If it is
    a dataframe, it is saved as a dataset. Dictionary keys must be strings and cannot contain the ``/`` character.

    Example usage:

    output_path = 'output_test.h5'
    dict_to_hdf(test_dict, output_path)
    test_file = pd.HDFStore(output_path)
    groups = test_file.keys()
    test_groups = test_file.get(groups[1])

    """
    new_hdf = pd.HDFStore(output_path)
    for key, item in test.items():
        logger.info(f'Key: {type(key)}')
        logger.info(f'Item: {type(item)}')
        if isinstance(item, dict):
            logger.info(f"Dict found. Running function with {h5_group+key} dict")
            dict_to_hdf(item, output_path, f'{h5_group}/{key}')
            logger.info(f"Function Successful with {h5_group}/{key}")

        elif isinstance(item, pd.DataFrame):
            logger.info(f"DF found. Saving dataframe with {h5_group}/{key} group.")
            new_hdf.put(f'{h5_group}/{key}', item)
        else:
            logger.info('Item detected was not a DataFrame. Please use alternate save format.')

    new_hdf.close()



# collect and test single ROIs
# input_path = 'C:/Users/Dezerae/Documents/Current Analysis/190327_YH_Machine learning optodrop classification/Python_results/Training_set/preprocessing/Ex2.h5'
# raw_data = pd.HDFStore(input_path)
# test = loader(raw_data, path='/')
# test.keys()
# rois = list(test['1']['single_cells']['after'].keys())
# for roi in rois:
#     plt.imshow(test['1']['single_cells']['after'][roi])
#     plt.colorbar()
#     plt.show()


# # test generated hdf file
# output_path = 'C:/Users/Dezerae/Documents/App_Dev_Projects/GEN_hdf_v_dict/output_test.h5'
# my_func(test, output_path)
# test = pd.HDFStore(output_path)
# groups = test.keys()
# test_groups = test.get(groups[1])
