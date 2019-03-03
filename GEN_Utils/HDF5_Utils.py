import numpy as np
import pandas as pd

def h5store(filename, key, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put(key, df)
    store.get_storer(key).attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

def dict_to_h5(filename, dictionary, **kwargs):
    store = pd.HDFStore(filename)
    for key, df in dictionary.items():
        store.put(key, df)
        store.get_storer(key).attrs.metadata = kwargs
    store.close()

# Example usage:
# a = pd.DataFrame(data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))
# filename = '/tmp/data.h5'
# metadata = dict(local_tz='US/Eastern')
# h5store(filename, a, **metadata)
# with pd.HDFStore(filename) as store:
#     data, metadata = h5load(store)
