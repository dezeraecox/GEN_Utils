import numpy as np
import pandas as pd

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

# Example usage:
# a = pd.DataFrame(data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))
# filename = '/tmp/data.h5'
# metadata = dict(local_tz='US/Eastern')
# h5store(filename, a, **metadata)
# with pd.HDFStore(filename) as store:
#     data, metadata = h5load(store)
