
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d

from mpsplines import (  # pylint: disable=import-error
    MeanPreservingInterpolation as MPI
)


def saveplot(n):

    grid_color = 'lavender'
    frame_color = '#aaaaee'

    ax = pl.gca()
    ax.xaxis.set_ticks(np.arange(0., 2*np.pi, np.pi/4), minor=False)
    ax.xaxis.set_ticks(np.arange(0., 2*np.pi, np.pi/24), minor=True)
    ax.yaxis.set_ticks(np.arange(-0.6, 1.21, 0.2), minor=False)
    ax.yaxis.set_ticks(np.arange(-0.6, 1.21, 0.1), minor=True)

    ax.xaxis.set_ticklabels(
        ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
         r'$\frac{3}{4}\pi$', r'$\pi$', r'$\frac{5}{4}\pi$',
         r'$\frac{3}{2}\pi$', r'$\frac{7}{4}\pi$'])

    ax.grid(which='both', lw=0.8, color=grid_color)

    for spine in ax.spines.values():
        spine.set_color(frame_color)
        spine.set_linewidth(0.8)

    ax.tick_params(which='both', width=0.8, color=frame_color,
                   direction='inout', right=True, top=True,
                   labelsize=11, labelcolor=frame_color)

    ax.set_xlabel('x', fontsize=12, color=frame_color)
    ax.set_ylabel('y', fontsize=12, color=frame_color)

    pl.legend(bbox_to_anchor=(0.5, 0.05), loc='lower center', ncol=2,
              fontsize=10, frameon=False)

    pl.axis([0, 2*np.pi, -0.6, 1.2])

    pl.tight_layout()
    pl.savefig(f'figure_{n:02d}.png')


n = 8

x = np.linspace(0., 2*np.pi, n*60)
y = (0.05 - np.sin(x))**2

xi = np.reshape(x, (n, 60)).mean(axis=1)
yi = np.reshape(y, (n, 60)).mean(axis=1)

pl.figure(figsize=(8, 3.5))

pl.plot(x, y, 'k--', lw=.8, label='unknown true process')
pl.plot(xi, yi, 'ko', ms=4.5, mfc='none', mew=1.5, mec='k',
        label='averaged known process')
saveplot(1)

sp2 = interp1d(xi, yi, kind=2, bounds_error=False, fill_value='extrapolate')
pl.plot(x, sp2(x), 'g-', lw=1, label='regular 2nd-order splines')
saveplot(2)

mpi = MPI(yi=yi, xi=xi)
pl.plot(x, mpi(x), 'r-', lw=1, label='mp-splines')
saveplot(3)

mpi = MPI(yi=yi, xi=xi, periodic=True)
pl.gca().lines[-1].remove()
pl.plot(x, mpi(x), 'r-', lw=1, label='mp-splines periodic')
saveplot(4)

mpi = MPI(yi=yi, xi=xi, periodic=True, min_val=0.05, cubic_window_size=3)
pl.gca().lines[-1].remove()
pl.plot(x, mpi(x), 'r-', lw=1, label='mp-splines periodic, min_val=0.05')
saveplot(5)
