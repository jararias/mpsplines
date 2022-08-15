
import subprocess

import numpy as np
import pylab as pl
from matplotlib.font_manager import FontProperties as fp
from scipy.interpolate import interp1d

from mpsplines import MeanPreservingInterpolation as MPI


def fuente(ttf):
    def inner(**kwargs):
        return fp(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')

np.set_printoptions(precision=8, linewidth=180)

ppi = 600
width = 170 / 25.4  # inches
height = 125 / 25.4  # inches

xi = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
yi = np.array([24, 25, 16, 17, 12, 5, 1, 1, 14, 27, 28, 30])
x = np.linspace(0.5, 12.5, 365)

pl.figure('figure02', figsize=(width, height), dpi=ppi)
pl.subplots_adjust(left=0.1, right=0.98, bottom=0.32, top=0.98)

pl.bar(xi, yi, 0.99, 0., color='0.6', edgecolor='blue',
       linewidth=0, alpha=0.3)

sp2 = interp1d(xi, yi, kind=2, fill_value='extrapolate')
pl.plot(x, sp2(x), ls='-', lw=0.8, color='dodgerblue', marker='',
        zorder=111, label=r'Regular 2$^\mathrm{nd}$-order splines')

sp3 = interp1d(xi, yi, kind=3, fill_value='extrapolate')
pl.plot(x, sp3(x), dashes=(6, 2), lw=0.8, color='dodgerblue', marker='',
        zorder=111, label=r'Regular 3$^\mathrm{rd}$-order splines')

mpi1 = MPI(xi, yi, periodic=True, min_val=None)
pl.plot(x, mpi1(x), lw=0.8, color='orange', ls='-', marker='',
        zorder=112, label='Mean-preserving splines')

mpi2 = MPI(xi, yi, periodic=True, min_val=0., cubic_window_size=5)
pl.plot(x, mpi2(x), lw=1.4, color='r', ls='-', marker='',
        zorder=112, label='Mean-preserving splines (locally relaxed)')

pl.plot([0.25, 12.75], [0, 0], marker='', ls='-', lw=0.5, color='black')

ax = pl.gca()
ax.spines['bottom'].set_position(('data', -12))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.set_ticks(np.arange(1, 13), minor=False)
ax.xaxis.set_ticklabels([
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
    'Sep', 'Oct', 'Nov', 'Dec'], fontproperties=arial(size=11))
ax.xaxis.set_tick_params(pad=6)
ax.set_xlabel('Month', labelpad=7, fontproperties=arial(size=13))
ax.set_xlim(0, 13)

ax.yaxis.set_ticks(np.arange(0, 31, 5), minor=False)
pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=12))
ax.set_ylabel('Accumulated Precipitation (mm)', labelpad=14,
              fontproperties=arial(size=14))
ax.set_ylim(-1, 33)

pl.legend(loc='upper left', bbox_to_anchor=(0.24, 0.98), prop=arial(size=9.2),
          frameon=True, facecolor='none', edgecolor='none')

ax = pl.gcf().add_axes([0.1, 0.13, 0.88, 0.15])


def monthly_deviation(interpolator):
    out = np.full(len(xi), np.nan)
    for k in range(len(xi)):
        x_ = np.linspace(xi[k]-0.5, xi[k]+0.5, 100)
        out[k] = np.trapz(interpolator(x_), x_) - yi[k]
    return out


kwargs = dict(align='edge', facecolor='white', linewidth=0.6)
ax.bar(xi - 0.5 + 0.35, monthly_deviation(sp2), 0.16, 0.,
       zorder=1000, edgecolor='dodgerblue', linestyle='-', **kwargs)
ax.bar(xi - 0.5 + 0.15, monthly_deviation(sp3), 0.16, 0.,
       zorder=1000, edgecolor='dodgerblue', linestyle='--', **kwargs)
ax.bar(xi - 0.5 + 0.55, monthly_deviation(mpi1), 0.16, 0.,
       zorder=1000, edgecolor='orange', linestyle='-', **kwargs)

kwargs['facecolor'] = 'red'
ax.bar(xi - 0.5 + 0.75, monthly_deviation(mpi2), 0.16, 0.,
       zorder=1000, edgecolor='red', linestyle='-', **kwargs)

ax.plot([0.25, 12.75], [0, 0], marker='', ls='-', lw=0.5,
        color='black', zorder=1001)

ax.set_yscale('symlog', linthresh=0.0001)

ax.spines['bottom'].set_position(('data', 0))
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)

pl.tick_params(axis='x', bottom=False, labelbottom=False)
ax.set_xlim(0, 13)

ax.yaxis.set_ticks((-1, -0.01, -0.0001, 0.0001, 0.01, 1), minor=False)
pl.setp(pl.getp(ax, 'yticklabels'), fontproperties=arial(size=8))
ax.tick_params(axis='y', width=0.5)
# ax.yaxis.grid(which='major', dashes=(8, 3), color='0.6', lw=0.4, zorder=-100)
for ygrid in (-1., -1e-2, -1e-4, 1e-4, 1e-2, 1):
    ax.plot([0.25, 12.75], [ygrid, ygrid], marker='', dashes=(8, 3),
            lw=0.4, color='0.6', zorder=-100)
ax.set_ylabel('Error (mm)', labelpad=8, fontproperties=arial(size=12))
ax.set_ylim(-2.5, 1)

pl.savefig('figure02_original.png', dpi=ppi)

subprocess.call(
    (f'convert figure02_original.png -resize {ppi*170/25.4} '
     'figure02_two_columns.png'.split())
)

subprocess.call(
    (f'convert figure02_original.png -resize {ppi*85/25.4} '
     'figure02_one_column.png'.split())
)
