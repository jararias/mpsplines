
import subprocess
from datetime import datetime

import numpy as np
import pylab as pl
import pandas as pd
from matplotlib.font_manager import FontProperties as fp
from matplotlib.dates import MonthLocator  # , DateFormatter

from loguru import logger
from scipy.interpolate import interp1d
from mpsplines import (
    MeanPreservingInterpolation as MPI
)


logger.enable('mpsplines')


def fuente(ttf):
    def inner(**kwargs):
        return fp(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')


def scipy_splines(series, times, **kwargs):
    dt64 = 'datetime64[ns]'
    xi = np.array(series.index, dtype=dt64).astype(np.float64)
    yi = series.to_numpy().ravel()
    x = np.array(times, dtype=dt64).astype(np.float64)
    kwargs.setdefault('kind', 2)
    kwargs.setdefault('fill_value', 'extrapolate')
    kwargs.setdefault('bounds_error', False)
    y = interp1d(xi, yi, **kwargs)(x)
    return pd.Series(data=y, index=times)


def read_data(variable):
    inp_fn = f'data/{variable}_hourly_merra2_2014.csv'
    data_hourly = pd.read_csv(inp_fn, index_col=0, parse_dates=True)
    data_hourly = data_hourly[data_hourly.index.year == 2014][variable]

    time_step = pd.Timedelta(1, 'D')
    data_daily = data_hourly.resample(time_step).mean()
    data_daily.index = data_daily.index + time_step / 2.

    return data_hourly, data_daily


ppi = 600
width = 170 / 25.4  # inches
height = 90 / 25.4  # inches

fig, axes = pl.subplots(nrows=2, ncols=1, figsize=(width, height),
                        dpi=600, sharex=True)

for n_var, variable in enumerate(('aod550', 'pwater')):

    data_hourly, data_daily = read_data(variable)

    time_step = pd.Timedelta(1, 'D')

    sp1_hourly = scipy_splines(data_daily, data_hourly.index, kind=1)
    sp1_daily = sp1_hourly.resample(time_step).mean()
    sp1_daily.index = sp1_daily.index + time_step / 2.

    sp2_hourly = scipy_splines(data_daily, data_hourly.index, kind=2)
    sp2_daily = sp2_hourly.resample(time_step).mean()
    sp2_daily.index = sp2_daily.index + time_step / 2.

    sp3_hourly = scipy_splines(data_daily, data_hourly.index, kind=3)
    sp3_daily = sp3_hourly.resample(time_step).mean()
    sp3_daily.index = sp3_daily.index + time_step / 2.

    mpi_hourly = pd.Series(
        data=MPI(data_daily.index, data_daily, min_val=0)(data_hourly.index),
        index=data_hourly.index, name=variable)
    mpi_daily = mpi_hourly.resample(time_step).mean()
    mpi_daily.index = mpi_daily.index + time_step / 2.

    ax = axes[n_var]

    ax.plot(data_hourly, marker='', ls='-', lw=.2, color='k',
            label='Reference data (hourly)')
    ax.plot(data_daily, marker='o', ls='', mec='k', ms=2,
            mfc='w', mew=0.3, label='Daily averages')
    ax.plot(sp2_hourly, marker='', ls='-', lw=.6, color='dodgerblue',
            label='Regular 2nd-order splines')
    ax.plot(mpi_hourly, marker='', ls='-', lw=.6, color='red',
            clip_on=False, label='Mean-preserving splines (min_val=0.)')

    mbd = sp2_hourly.sub(data_hourly).mean()
    rmsd = sp2_hourly.sub(data_hourly).pow(2).mean()**0.5
    unit = 'cm' if variable == 'pwater' else ''
    ax.text(0.02, 0.81, f'MBD={mbd: 6.3f} {unit}   RMSD={rmsd:5.3f} {unit}',
            transform=ax.transAxes, ha='left', va='center', color='dodgerblue',
            fontproperties=arial(size=5))

    mbd = mpi_hourly.sub(data_hourly).mean()
    rmsd = mpi_hourly.sub(data_hourly).pow(2).mean()**0.5
    ax.text(0.02, 0.74, f'MBD={mbd: 6.3f} {unit}   RMSD={rmsd:5.3f} {unit}',
            transform=ax.transAxes, ha='left', va='center', color='red',
            fontproperties=arial(size=5))

    ax.spines['bottom'].set_position(
        ('data', -0.1/(1. if variable == 'pwater' else 6.0)))
    ax.spines['left'].set_position(
        ('data', pl.date2num(datetime(2013, 12, 29))))
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.tick_params(width=0.5)

    if variable == 'pwater':
        ax.yaxis.set_ticks(np.arange(0., 6.0, 1.2))
    pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=7))
    ylabel = ('Precipitable water (cm)' if variable == 'pwater'
              else 'AOD at 550 nm')
    ax.set_ylabel(ylabel, labelpad=12 if variable == 'pwater' else 7,
                  fontproperties=arial(size=8))
    ax.set_ylim(0, 6.0 if variable == 'pwater' else 1.0)

    ax.xaxis.set_major_locator(
        MonthLocator(bymonth=range(2, 13, 2), bymonthday=1))
    ax.xaxis.set_minor_locator(
        MonthLocator(bymonth=range(1, 12, 2), bymonthday=1))
    pl.setp(pl.getp(ax, 'xticklabels'), fontproperties=arial(size=7))
    if n_var == 1:
        ax.set_xlabel('Date, year 2014', labelpad=4,
                      fontproperties=arial(size=8))
    ax.set_xlim(datetime(2014, 1, 1, 0), datetime(2015, 1, 1, 0))

    ax.legend(loc='upper left', ncol=4, bbox_to_anchor=(0.01, 1.06),
              prop=arial(size=6), frameon=True, facecolor='none',
              edgecolor='none')

    # stats binned by daily AOD...
    interp_hourly = pd.DataFrame(
        data={'sp1': sp1_hourly, 'sp2': sp2_hourly,
              'sp3': sp3_hourly, 'mpi': mpi_hourly},
        index=data_hourly.index
    )

    data_daily_hourly = data_daily.copy()
    data_daily_hourly.index = data_daily_hourly.index.map(
        lambda t: t.replace(hour=0, minute=30))
    data_daily_hourly = data_daily_hourly.reindex(
        data_hourly.index, method='ffill')

    models = ('sp1', 'sp2', 'sp3', 'mpi')
    print(
        '    '.join(
            [f'{variable.upper():<14}', f'{"MBD":_^30}',
             f'{"STD":_^26}', f'{"RMSD":_^26}']
        )
    )
    print(
        '    '.join(
            [f'{"Interval":<14}',
             '  '.join([f'{model:>6}' for model in models]),
             '  '.join([f'{model:>5}' for model in models]),
             '  '.join([f'{model:>5}' for model in models])]
        )
    )
    bins = pd.qcut(data_daily_hourly, q=10)
    for the_bin, subset in interp_hourly.groupby(bins):
        mbd = subset.apply(lambda x: (x-data_hourly).mean())
        std = subset.apply(lambda x: (x-data_hourly).std())
        rmsd = subset.apply(lambda x: (x-data_hourly).pow(2).mean()**0.5)
        row = [f'({the_bin.left:5.3f}, {the_bin.right:5.3f}]']
        row.append('  '.join([f'{mbd[model]: 5.3f}' for model in models]))
        row.append('  '.join([f'{std[model]:5.3f}' for model in models]))
        row.append('  '.join([f'{rmsd[model]:5.3f}' for model in models]))
        print('    '.join(row))

    bins = pd.qcut(data_daily_hourly, q=1)
    for the_bin, subset in interp_hourly.groupby(bins):
        mbd = subset.apply(lambda x: (x-data_hourly).mean())
        std = subset.apply(lambda x: (x-data_hourly).std())
        rmsd = subset.apply(lambda x: (x-data_hourly).pow(2).mean()**0.5)
        row = [f'({the_bin.left:5.3f}, {the_bin.right:5.3f}]']
        row.append('  '.join([f'{mbd[model]: 5.3f}' for model in models]))
        row.append('  '.join([f'{std[model]:5.3f}' for model in models]))
        row.append('  '.join([f'{rmsd[model]:5.3f}' for model in models]))
        print('    '.join(row))

pl.tight_layout(h_pad=1)
pl.savefig('figure04_original.png', dpi=ppi)

subprocess.call(
    (f'convert figure04_original.png -resize {ppi*170/25.4} '
     'figure04_two_columns.png'.split())
)

subprocess.call(
    (f'convert figure04_original.png -resize {ppi*85/25.4} '
     'figure04_one_column.png'.split())
)
