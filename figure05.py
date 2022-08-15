
import subprocess
from datetime import datetime

import numpy as np
import pylab as pl
import pandas as pd
from matplotlib.font_manager import FontProperties as fp
from matplotlib.dates import MonthLocator
from scipy.interpolate import interp1d

from mpsplines import (
    MeanPreservingInterpolation as MPI,
    MeanPreservingMonthlyLTAInterpolation as MPIlta
)


def fuente(ttf):
    def inner(**kwargs):
        return fp(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')


def read_data(site):
    data_lta = pd.read_csv('data/ground_albedo_lta_merra2.csv', index_col=0)
    print(data_lta.columns)
    data_lta = data_lta[site]
    return data_lta


def get_day_of_year(times):
    jan_1st = times.astype('datetime64[Y]').astype('datetime64[ns]')
    deltas = times.astype('datetime64[ns]') - jan_1st
    doys = deltas.astype('timedelta64[D]') + np.timedelta64(1, 'D')
    return doys.astype(np.float64)


def get_number_of_days_in_year(times):
    one_day = np.timedelta64(1, 'D')
    one_year = np.timedelta64(1, 'Y')
    dec_31st = times.astype('datetime64[Y]') + one_year - one_day
    return get_day_of_year(dec_31st)


def scipy_splines(series, times, **kwargs):
    dt64 = 'datetime64[ns]'
    lta_times = pd.date_range('2020-01', periods=12, freq='MS').map(
        lambda t: t.replace(day=15, hour=12, minute=0))
    days = np.array(lta_times, dtype=dt64)
    xi = get_day_of_year(days) / get_number_of_days_in_year(days)
    yi = series.to_numpy().ravel()
    x = np.array(times, ndmin=1, dtype=dt64)
    x = get_day_of_year(x) / get_number_of_days_in_year(x)
    kwargs.setdefault('kind', 2)
    kwargs.setdefault('fill_value', 'extrapolate')
    kwargs.setdefault('bounds_error', False)
    y = interp1d(xi, yi, **kwargs)(x)
    return pd.Series(data=y, index=times)


# site = 'Alice Springs'
# site = 'Boulder'
# site = 'Desert Rock'
site = 'Gobabeb'
# site = 'Lerwick'
# site = 'Tamanrasset'
# site = 'Sonnblick'
# site = 'Solar Village'
# site = 'Toravere'
# site = 'Tateno'
# site = 'Rock Springs'
# site = 'Goodwin Creek'

ppi = 600
width = 170/25.4  # inches
height = 55/25.4  # inches


data_lta = read_data(site)
print(data_lta)

times_daily = pd.date_range('2022-01-01', '2025-01-01', freq='D')

times_2023_monthly = pd.date_range('2023-01', periods=12, freq='MS').map(
    lambda t: t.replace(day=15, hour=12, minute=0))

times_2023_daily = pd.date_range('2023-01-01', '2023-12-31', freq='D')

pl.figure('figure05', figsize=(width, height), dpi=600)

pl.plot(pd.DataFrame(data=data_lta.to_numpy(), index=times_2023_monthly),
        ls='', marker='o', ms=3, mec='k', mfc='w', mew=0.5,
        zorder=101, label='Long-term monthly average ground albedo')

pl.plot(scipy_splines(data_lta, times_2023_daily, kind=2),
        marker='', ls='-', lw=0.6, color='dodgerblue',
        zorder=102, label=r'Regular 2nd-order splines')

mpi = MPI(times_2023_monthly, data_lta, periodic=True, min_val=0.)
pl.plot(pd.Series(data=mpi(times_2023_daily), index=times_2023_daily),
        marker='', ls='-', lw=0.8, color='red', zorder=104,
        label='Mean-preserving splines with periodic boundaries')

mpi = MPIlta(data_lta, min_val=0.)
pl.plot(pd.Series(data=mpi(times_daily), index=times_daily),
        marker='', ls='-', lw=0.8, color='0.8', zorder=103,
        label='Mean-preserving splines with LTA extrapolation')

ax = pl.gca()

ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_position(
    ('data', pl.date2num(datetime(2021, 12, 26))))
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.set_major_locator(
    MonthLocator(bymonth=(1, 6), bymonthday=1))
ax.xaxis.set_minor_locator(
    MonthLocator(bymonth=np.arange(1, 13, 1), bymonthday=1))
ax.set_xlim(datetime(2022, 1, 1, 0), datetime(2024, 12, 31, 0))

if site == 'Gobabeb':
    ax.yaxis.set_ticks(np.arange(0.306, 0.331, 0.004), minor=False)
    ax.yaxis.set_ticks(np.arange(0.306, 0.333, 0.001), minor=True)
    ax.set_ylim(0.306, 0.333)

ymin = ax.get_ylim()[0]*0.999
ax.spines['bottom'].set_position(('data', ymin))  # 0.3057))

ax.tick_params(which='major', length=3.5, width=0.5)
ax.tick_params(which='minor', length=1.5, width=0.5)

pl.setp(pl.getp(ax, 'xticklabels'), fontproperties=arial(size=7))
pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=7))
ax.set_ylabel('Ground albedo', labelpad=7, fontproperties=arial(size=8))
ax.set_xlabel('Date', labelpad=3, fontproperties=arial(size=8))

leg = ax.legend(loc='upper right', bbox_to_anchor=(1., 0.95), ncol=1,
                prop=arial(size=5.5), frameon=True, facecolor='white',
                edgecolor='none')
leg.set_zorder(1001)

pl.tight_layout(rect=(0, 0, 1, 1))
pl.savefig('figure05_original.png', dpi=ppi)

subprocess.call(
    (f'convert figure05_original.png -resize {ppi*170/25.4} '
     'figure05_two_columns.png'.split())
)

subprocess.call(
    (f'convert figure05_original.png -resize {ppi*85/25.4} '
     'figure05_one_column.png'.split())
)
