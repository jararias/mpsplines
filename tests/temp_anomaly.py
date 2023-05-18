
# pylint: disable=import-error

import numpy as np
import pandas as pd
import pylab as pl
from scipy.interpolate import interp1d

from mpsplines import MeanPreservingInterpolation as MPI


fname = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
# fname = 'GLB.Ts+dSST.csv'

data = pd.read_csv(fname, skiprows=1, index_col=0,
                   usecols=range(13), na_values='***')

data = data.stack()
timestamps = data.index.get_level_values(0).astype(str) + data.index.get_level_values(1)
data.index = pd.to_datetime(timestamps, format='%Y%b') + pd.Timedelta(14, 'D')
data.plot(alpha=0.2, label=f'original monthly ({data.mean()=:.3f})')

data_yearly = data.groupby(data.index.year).mean()
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y') + pd.Timedelta(180, 'D')
data_yearly = data_yearly[:-1]  # drop the last year, because it is not full
idata = pd.Series(
    index=data.index,
    data=interp1d(data_yearly.index.to_numpy().astype(np.float64), data_yearly.values,
                  kind='cubic', bounds_error=False)(data.index.to_numpy().astype(np.float64)))
idata.plot(label=f'scipy cubic ({idata.mean()=:.3f})')

data_yearly = data.groupby(data.index.year).mean()
data_yearly.index = pd.to_datetime(data_yearly.index, format='%Y') + pd.Timedelta(180, 'D')
data_yearly = data_yearly[:-1]  # drop the last year, because it is not full

years = data_yearly.index.year.unique()
x_edges = pd.to_datetime([f'{y}-01-01' for y in range(years.min(), years.max()+2)])

mpi = MPI(yi=data_yearly.values, x_edges=x_edges)  # , xi=data_yearly.index)  #
idata = pd.Series(data=mpi(data.index), index=data.index)
idata.plot(label=f'mpsplines ({idata.mean()=:.3f})')
print(idata.mean())

pl.legend()
pl.show()
