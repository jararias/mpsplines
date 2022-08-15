
import subprocess

import numpy as np
from matplotlib import rcParams
import pylab as pl
from matplotlib.font_manager import FontProperties as fp
from scipy.interpolate import interp1d
from scipy.optimize import root

from mpsplines import (
    MeanPreservingInterpolation as MPI
)


def fuente(ttf):
    def inner(**kwargs):
        return fp(fname=ttf, **kwargs)
    return inner


arial = fuente('fonts/arial.ttf')
arial_bold = fuente('fonts/arialbd.ttf')

rcParams['hatch.linewidth'] = 0.6


xi = np.array([1, 2, 3])
yi = np.array([.25, .8, .35])
x = np.linspace(0.45, 3.55, 1000)

ppi = 600
width = 170 / 25.4  # inches
height = 125 / 25.4  # inches

pl.figure('figure01', figsize=(width, height), dpi=ppi)
pl.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)

# plot interpolation nodes
pl.plot(xi, yi, ls='', marker='o', mec='k', mew=1.0, mfc='w',
        zorder=113, label='Interpolation nodes')

# plot 2nd-order regular splines
sp2 = interp1d(xi, yi, kind=2, fill_value='extrapolate')(x)
sp2[sp2 < 0.03] = np.nan
pl.plot(x, sp2, dashes=(8, 0), lw=1.2, color='dodgerblue', marker='',
        zorder=111, label=r'Regular 2nd-order splines')

# mp-splines
mpi = MPI(xi, yi, periodic=True)

kwargs = dict(lw=0.2, color='k', marker='', dashes=(8, 4))
for x_ in (0.5, 1.5, 2.5):
    mean = np.mean(mpi(x)[(x >= x_) & (x <= (x_ + 1))])
    pl.plot([x_, x_], [0, 0.95], **kwargs)
    pl.plot([(x_+1), (x_+1)], [0, 0.95], **kwargs)

# pl.plot(x+0.004, mpi(x)-0.003, lw=1.5, alpha=0.9,
#         zorder=112, color='darkred', ls='-', marker='')
pl.plot(x, mpi(x), lw=1.2, color='r', ls='-', marker='',
        zorder=112, label='Mean-preserving splines')

# shading
cunder = '#ffe0ee'  # '#fdb768'
cover = '#feedc2'
kwargs = {'fill': True, 'alpha': 1., 'ls': '-', 'lw': 0.}

r = root(lambda x: mpi(x) - yi[0], 1).x
x = np.linspace(0.5, r, 50)
y = np.r_[mpi(x), np.full(50, yi[0])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cover, **kwargs)
pol.set_linewidth(0.6)
x = np.linspace(r, 1.5, 50)
y = np.r_[np.full(50, yi[0]), mpi(x[::-1])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cunder, **kwargs)
pol.set_linewidth(0.6)

r1 = root(lambda x: mpi(x) - yi[1], 1.6).x
x = np.linspace(1.5, r1, 50)
y = np.r_[mpi(x), np.full(50, yi[1])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cover, **kwargs)
pol.set_linewidth(0.6)
r2 = root(lambda x: mpi(x) - yi[1], 2.4).x
x = np.linspace(r2, 2.5, 50)
y = np.r_[mpi(x), np.full(50, yi[1])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cover, **kwargs)
pol.set_linewidth(0.6)
x = np.linspace(r1, r2, 50)
y = np.r_[np.full(50, yi[1]), mpi(x[::-1])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cunder, **kwargs)
pol.set_linewidth(0.6)

r = root(lambda x: mpi(x) - yi[2], 2.9).x
x = np.linspace(2.5, r, 50)
y = np.r_[np.full(50, yi[2]), mpi(x[::-1])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cunder, **kwargs)
pol.set_linewidth(0.6)
x = np.linspace(r, 3.5, 50)
y = np.r_[mpi(x), np.full(50, yi[2])]
pol, = pl.fill(np.r_[x, x[::-1]], y, fc=cover, **kwargs)
pol.set_linewidth(0.6)


# arrows
kwargs = dict(head_width=0.007, head_length=0.05, length_includes_head=True)
pl.arrow(0.5, 0.92, 1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.arrow(1.5, 0.92, -1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.text(1., 0.935, r'Spline for the 1-st interpolation node',
        ha='center', va='center', fontproperties=arial(size=7))

pl.arrow(1.5, 0.92, 1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.arrow(2.5, 0.92, -1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.text(2., 0.935, r'Spline for the 2-nd interpolation node',
        ha='center', va='center', fontproperties=arial(size=7))

pl.arrow(2.5, 0.92, 1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.arrow(3.5, 0.92, -1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.text(3., 0.935, r'Spline for the 3-rd interpolation node',
        ha='center', va='center', fontproperties=arial(size=7))

pl.arrow(1.5, 0.1, 1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.arrow(2.5, 0.1, -1., 0., color='k', lw=0.2, zorder=1001, **kwargs)
pl.text(2., 0.105, 'Averaging window for\nthe 2-nd interpolation node',
        va='bottom', multialignment='center', ha='center',
        fontproperties=arial(size=7))

pl.arrow(0.9, 0.5, 0.43, -0.18, head_width=0.01, head_length=0.02,
         color='k', lw=0.3, zorder=1001)
pl.arrow(0.8, 0.5, -0.05, -0.3, head_width=0.014,
         head_length=0.018, color='k', lw=0.3, zorder=1001)
pl.text(0.85, 0.52, 'Areas of equal size', ha='center', va='center',
        fontproperties=arial(size=8))

# labels
ax = pl.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.xaxis.set_ticks([1, 2, 3], minor=False)
ax.xaxis.set_ticklabels([r'x$_1$', r'x$_2$', r'x$_3$'], minor=False,
                        fontproperties=arial_bold(size=13))
ax.xaxis.set_ticks([0.5, 1.5, 2.5, 3.5], minor=True)
kwargs = dict(va='top', fontproperties=arial(size=10))
ax.text(0.52, -0.01, r'x$_{\ell,1}$', ha='left', **kwargs)
ax.text(1.48, -0.01, r'x$_{\mathrm{{u}},1}$', ha='right', **kwargs)
ax.text(1.52, -0.01, r'x$_{\ell,2}$', ha='left', **kwargs)
ax.text(2.48, -0.01, r'x$_{\mathrm{{u}},2}$', ha='right', **kwargs)
ax.text(2.52, -0.01, r'x$_{\ell,3}$', ha='left', **kwargs)
ax.text(3.48, -0.01, r'x$_{\mathrm{{u}},3}$', ha='right', **kwargs)
ax.set_xlabel('Independent variable', labelpad=10,
              fontproperties=arial(size=14))

ax.yaxis.set_ticks(yi, minor=False)
ax.yaxis.set_ticklabels([r'y$_1$', r'y$_2$', r'y$_3$'], minor=False,
                        fontproperties=arial_bold(size=13))
ax.set_ylabel('Interpolated variable', labelpad=14,
              fontproperties=arial(size=14))
ax.set_ylim(0, 1)

pl.legend(loc='upper left', bbox_to_anchor=(0.64, 0.88),
          prop=arial(size=8.2), frameon=True, facecolor='none',
          edgecolor='none')

pl.savefig('figure01_original.png', dpi=ppi)

subprocess.call(
    (f'convert figure01_original.png -resize {ppi*170/25.4} '
     'figure01_two_columns.png'.split())
)

subprocess.call(
    (f'convert figure01_original.png -resize {ppi*85/25.4} '
     'figure01_one_column.png'.split())
)
