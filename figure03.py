
import datetime
import subprocess

import numpy as np
import pylab as pl
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties as fp
from matplotlib.dates import DateFormatter, HourLocator

from mpsplines import (
    MeanPreservingInterpolation as MPI
)


def fuente(ttf):
    def inner(**kwargs):
        return fp(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')

ppi = 600
width = 90 / 25.4  # inches
height = 66 / 25.4  # inches


# high-res data
data = pd.read_csv('data/ghics_profile.csv', index_col=0, parse_dates=True)
ghi_1min = data['ghi_W/m2']
eth_1min = data['eth_W/m2']


def upscale(series, res):
    time_step = pd.Timedelta(res)
    new_series = series.resample(time_step).mean()
    new_series.index = new_series.index + time_step / 2.
    return new_series


def downscale(ghi_lowres, method, **kwargs):

    dt64 = 'datetime64[ns]'

    if method == 'regular_splines':
        t_inp = np.array(ghi_lowres.index, dtype=dt64).astype(np.float64)
        t_out = np.array(ghi_1min.index, dtype=dt64).astype(np.float64)
        kwargs.setdefault('bounds_error', False)
        values = interp1d(t_inp, ghi_lowres, **kwargs)(t_out)
        values[eth_1min <= 0.] = np.nan
        return pd.Series(data=values, index=ghi_1min.index)

    if method == 'mp_splines':
        # mpsplines "understand" datetime-like data...
        values = MPI(ghi_lowres.index, ghi_lowres, **kwargs)(ghi_1min.index)
        values[eth_1min <= 0.] = np.nan
        return pd.Series(data=values, index=ghi_1min.index)

    if method == 'Kt_space':
        eth_lowres = kwargs.pop('eth')
        Kt_lowres = ghi_lowres.divide(eth_lowres).where(eth_lowres > 0., 0.)
        t_lowres = np.array(Kt_lowres.index, dtype=dt64).astype(np.float64)
        t_highres = np.array(ghi_1min.index, dtype=dt64).astype(np.float64)
        kwargs.setdefault('bounds_error', False)
        kwargs.setdefault('kind', 1)
        values = interp1d(t_lowres, Kt_lowres, **kwargs)(t_highres)
        values[ghi_1min <= 0.] = np.nan
        return pd.Series(data=values*eth_1min, index=ghi_1min.index)


# figure...

time_step_lowres = '2H'
ghi_lowres = upscale(ghi_1min, time_step_lowres)
eth_lowres = upscale(eth_1min, time_step_lowres)

pl.figure('figure03', figsize=(width, height), dpi=ppi)
pl.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.98)

times = ghi_1min.index.time
universe = (times > datetime.time(2, 30)) & (times < datetime.time(21, 30))
pl.plot(ghi_1min[universe], marker='', ls='-', lw=1.2, color='k',
        label='1-min reference data', clip_on=False)

times = ghi_lowres.index.time
universe = (times > datetime.time(2, 30)) & (times < datetime.time(21, 30))
pl.plot(ghi_lowres[universe], marker='o', ls='', ms=3, mec='k', mfc='w',
        mew=0.6, label='2-hour averaged data', clip_on=False)

ghi_sp2_1min = downscale(ghi_lowres, 'regular_splines', kind=2)
pl.plot(ghi_sp2_1min, marker='', ls='-', lw=0.7, color='dodgerblue',
        label=r'Regular 2$^\mathrm{nd}$-order splines')

ghi_Kti_1min = downscale(ghi_lowres, 'Kt_space', eth=eth_lowres)
pl.plot(ghi_Kti_1min, marker='', ls='-', lw=0.7, color='orange',
        label='Linear in KT space')

ghi_mpi_1min = downscale(ghi_lowres, 'mp_splines', min_val=None)
pl.plot(ghi_mpi_1min, marker='', ls='-', lw=0.7, color='r',
        label='Mean-preserving splines')

combine = datetime.datetime.combine

ax = pl.gca()
ax.spines['bottom'].set_position(('data', -23))
pos = combine(ghi_1min.index.date[0], datetime.time(2, 0))
ax.spines['left'].set_position(('data', pl.date2num(pos)))
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

pl.tick_params(width=0.5, pad=2)

ax.xaxis.set_major_locator(HourLocator(byhour=(4, 8, 12, 16, 20)))
ax.xaxis.set_major_formatter(DateFormatter(fmt='%H:%M'))
ax.xaxis.set_minor_locator(
    HourLocator(byhour=(3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21)))
pl.setp(pl.getp(ax, 'xticklabels'), fontproperties=arial(size=6))
ax.set_xlabel('Universal Time on 2022-06-15', labelpad=4,
              fontproperties=arial(size=7))
ax.set_xlim(
    combine(ghi_1min.index.date[0], datetime.time(2, 15)),
    combine(ghi_1min.index.date[0], datetime.time(21, 45))
)

ax.yaxis.set_ticks(np.arange(0, 1001, 200), minor=False)
pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=6))
ax.set_ylabel('Solar Irradiance (W/m$^2$)', labelpad=3,
              fontproperties=arial(size=7))
ax.set_ylim(0, 1060)

pl.legend(loc='upper left', bbox_to_anchor=(0., 1.), prop=arial(size=5),
          frameon=True, facecolor='none', edgecolor='none')

pl.savefig('figure03_original.png', dpi=ppi)

subprocess.call(
    (f'convert figure03_original.png -resize {ppi*170/25.4} '
     'figure03_two_columns.png'.split())
)

subprocess.call(
    (f'convert figure03_original.png -resize {ppi*85/25.4} '
     'figure03_one_column.png'.split())
)


# table figures...

def scores(mod, obs):
    residue = mod - obs
    residue = residue[eth_1min > 0]
    return {
        'mbd': np.mean(residue),
        'mad': np.mean(np.abs(residue)),
        'rmsd': np.sqrt(np.mean(residue**2))
    }


tables = {
    score_name: '      SP2   SP3    KT   MPI\n'
    for score_name in ('mbd', 'mad', 'rmsd')
}

for coarse_res in ('1H', '2H', '3H', '4H'):
    ghi_lowres = upscale(ghi_1min, coarse_res)
    eth_lowres = upscale(eth_1min, coarse_res)
    sp2 = downscale(ghi_lowres, 'regular_splines', kind=2)
    for score_name, score_value in scores(sp2, ghi_1min).items():
        tables[score_name] += f'{coarse_res:<3}'
        tables[score_name] += f'  {score_value:4.1f}'
    sp3 = downscale(ghi_lowres, 'regular_splines', kind=3)
    for score_name, score_value in scores(sp3, ghi_1min).items():
        tables[score_name] += f'  {score_value:4.1f}'
    Kti = downscale(ghi_lowres, 'Kt_space', eth=eth_lowres)
    for score_name, score_value in scores(Kti, ghi_1min).items():
        tables[score_name] += f'  {score_value:4.1f}'
    mpi = downscale(ghi_lowres, 'mp_splines', min_val=None)
    for score_name, score_value in scores(mpi, ghi_1min).items():
        tables[score_name] += f'  {score_value:4.1f}'
        tables[score_name] += '\n'

for table_score, table in tables.items():
    print(f'{table_score.upper()} (W/m2):\n{table}')
