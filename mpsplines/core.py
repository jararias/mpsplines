"""
Mean-preserving splines interpolation

Author: Jose A Ruiz-Arias, University of Malaga (Spain)
e-mail: jararias@uma.es

The interpolation is performed with the following classes:

class MeanPreservingInterpolation

class MeanPreservingLTAInterpolation

See inline documentation for use explanations.

You can run also:

python -m doctest mpsplines.py

for a demo

"""
from loguru import logger
from datetime import datetime

import numpy as np
from scipy import optimize

try:
    from scipy.sparse import csc_matrix, csr_matrix, linalg
    USE_SPARSE_MATRICES = True
except ImportError:
    from scipy import linalg
    USE_SPARSE_MATRICES = False

    
logger.disable(__name__)


def find_minimum(p, dxl, dxu):
    poly = np.poly1d(p)
    derivative = np.poly1d(np.polyder(p, 1))
    for root in filter(np.isrealobj, np.roots(derivative)):
        if (root >= dxl) and (root <= dxu):
            return min(poly(dxl), poly(dxu), poly(root))
    return min(poly(dxl), poly(dxu))


def second_order_unconstrained_interpolation(xi, yi, x_edges, periodic=False):
    '''
    Compute the coefficients of the mean-preserving 2nd-order splines
    by solving a system of linear equations
    '''

    order = 2
    n_coefs = order + 1
    n_samples = len(xi)

    Dx = np.diff(x_edges)
    Dxl = x_edges[:-1] - xi
    Dxu = x_edges[1:] - xi

    # There is one 2nd-order spline for each input datum and three
    # coefficients for each spline. Thus, a total of n_coefs * n_samples
    # coefficients must be calculated. The column matrix of coefficients,
    # X, must verify that:
    #   A X = B
    # where A is a square matrix of size n_coefs * n_samples and B is a
    # column matrix of size n_coefs * n_samples. This system of equations
    # renders the continuity of the splines and the preservation of the mean
    A = np.zeros((n_samples * n_coefs, n_samples * n_coefs), dtype=np.float64)
    B = np.zeros(n_samples * n_coefs, dtype=np.float64)

    Dx2 = Dx**2
    Dxu2 = Dxu**2
    Dxl2 = Dxl**2

    # conditions to preserve the mean...
    for i in range(n_samples):
        A[i, i*n_coefs:(i+1)*n_coefs] = [
            Dx2[i]/3. + Dxl[i]*Dxu[i], Dx[i]/2. + Dxl[i], 1.]
    B[:n_samples] = yi

    # conditions to preserve the splines continuity...
    for i in range(n_samples - 1):
        A[i+n_samples, i*n_coefs:(i+2)*n_coefs] = [
            Dxu2[i], Dxu[i], 1., -Dxl2[i+1], -Dxl[i+1], -1.
        ]

    # conditions to preserve the first derivative continuity...
    for i in range(n_samples - 1):
        A[i+2*n_samples, i*n_coefs:(i+2)*n_coefs] = [
            2*Dxu[i], 1., 0., -2*Dxl[i+1], -1, 0.
        ]

    if periodic is True:
        # conditions to preserve the splines continuity...
        A[2*n_samples-1, -n_coefs:] = [Dxu2[-1], Dxu[-1], 1.]
        A[2*n_samples-1, :n_coefs] = [-Dxl2[0], -Dxl[0], -1.]
        # conditions to preserve the first derivative continuity...
        A[3*n_samples-1, -n_coefs:] = [2*Dxu[-1], 1., 0.]
        A[3*n_samples-1, :n_coefs] = [-2*Dxl[0], -1., 0.]
    else:
        # assume the second derivative is conserved in the two
        # leftmost and rightmost splines
        A[2*n_samples-1, :2*n_coefs] = [1., 0., 0., -1., 0., 0.]
        A[3*n_samples-1, -2*n_coefs:] = [1., 0., 0., -1., 0., 0.]

    if USE_SPARSE_MATRICES is True:  # much faster
        As = csc_matrix(A, dtype=np.float64)
        Bs = csr_matrix(B, dtype=np.float64).T
        return np.reshape(linalg.spsolve(As, Bs), (n_samples, n_coefs))
    else:
        return np.reshape(linalg.solve(A, B), (n_samples, n_coefs))


def third_order_constrained_interpolation(
        xi, yi, x_edges, min_val, p_guess, force_second_order=[]):
    '''
    Compute the coefficients of the mean-preserving 3rd-order splines
    by minimizing the square error to find an interpolation curve which
    is always greater or equal than min_val
    '''

    order = 3
    n_coefs = order + 1
    n_samples = len(xi)

    Dx = np.diff(x_edges)
    Dxl = x_edges[:-1] - xi
    Dxu = x_edges[1:] - xi

    # initial coefficient guess
    G = p_guess
    # arrays used to preserve the continuity of the splines
    Us = np.stack([Dxu**3, Dxu**2, Dxu, np.ones(n_samples)])
    Ls = np.stack([Dxl**3, Dxl**2, Dxl, np.ones(n_samples)])
    # arrays used to preserve the continuity of the splines derivatives
    Ud = np.stack([3*Dxu**2, 2*Dxu, np.ones(n_samples)])
    Ld = np.stack([3*Dxl**2, 2*Dxl, np.ones(n_samples)])
    # arrays used to preserve the mean
    Um = np.stack([(Dxu**4)/4., (Dxu**3)/3., (Dxu**2)/2., Dxu])
    Lm = np.stack([(Dxl**4)/4., (Dxl**3)/3., (Dxl**2)/2., Dxl])
    Dm = Um - Lm
    Dr = Dm/Dx[None, :]

    third_order_splines = [
        k for k in range(n_samples) if k not in force_second_order
    ]

    for e in third_order_splines:
        for k in range(-1, 1 + 1):
            if (e + k) in force_second_order:
                force_second_order.pop(force_second_order.index(e + k))

    def _fit_func(p, x, force_second_order=[]):
        P = np.reshape(p, (n_samples, n_coefs))
        P[force_second_order, 0] = 0.
        y = np.empty_like(x)
        ind = np.clip(np.digitize(x, x_edges)-1, 0, n_samples-1)
        for i in range(n_samples):
            universe = ind == i
            dx = x[universe] - xi[i]
            y[universe] = (
                P[i, 0]*(dx**3) + P[i, 1]*(dx**2) + P[i, 2]*dx + P[i, 3]
            )
        return y

    def _preserve_continuity(p):
        constraints = np.zeros(2 * (n_samples+1))
        P = np.reshape(p, (n_samples, n_coefs))
        # continuity of the splines
        constraints[0] = (  # left bound
            np.dot(G[0, :], Ls[:, 0]) - np.dot(P[0, :], Ls[:, 0]))
        constraints[1:n_samples] = (  # internal bounds
            np.diagonal(np.dot(P[:-1, :], Us[:, :-1])) -
            np.diagonal(np.dot(P[1:, :], Ls[:, 1:])))
        constraints[n_samples] = (  # right bound
            np.dot(P[-1, :], Us[:, -1]) - np.dot(G[-1, :], Us[:, -1]))
        # continuity of the derivative
        constraints[n_samples+1] = (  # left bound
            np.dot(G[0, :-1], Ld[:, 0]) - np.dot(P[0, :-1], Ld[:, 0]))
        constraints[n_samples+2:2*n_samples+1] = (  # internal bounds
            np.diagonal(np.dot(P[:-1, :-1], Ud[:, :-1])) -
            np.diagonal(np.dot(P[1:, :-1], Ld[:, 1:])))
        constraints[2*n_samples+1] = (  # right bound
            np.dot(P[-1, :-1], Ud[:, -1]) - np.dot(G[-1, :-1], Ud[:, -1]))
        return constraints

    def _continuity_jac(p):
        jac = np.zeros((2 * (n_samples+1), len(p)))
        rows = np.repeat(np.arange(2*(n_samples+1))[:, None], n_coefs, axis=1)
        cols = np.reshape(np.arange(len(p)), (n_samples, n_coefs))
        # jacobian for the continuity of the splines
        jac[0, :n_coefs] = -Ls[:, 0]  # left bound
        jac[rows[1:n_samples, :], cols[:-1, :]] = Us[:, :-1].T
        jac[rows[1:n_samples, :], cols[1:, :]] = -Ls[:, 1:].T
        jac[n_samples, -n_coefs:] = Us[:, -1]  # right bound
        # jacobian for the continuity of the derivative
        jac[n_samples+1, :n_coefs-1] = -Ld[:, 0]  # left bound
        jac[rows[n_samples+2:2*n_samples+1, :-1], cols[:-1, :-1]] = Ud[:, :-1].T  # noqa
        jac[rows[n_samples+2:2*n_samples+1, :-1], cols[1:, :-1]] = -Ld[:, 1:].T
        jac[2*n_samples+1, -n_coefs:-1] = Ud[:, -1]  # right bound
        return jac

    def _preserve_mean(p):
        P = np.reshape(p, (n_samples, n_coefs))
        return np.diagonal(np.dot(P, Dr)) - yi

    def _mean_jac(p):
        jac = np.zeros((n_samples, len(p)))
        rows = np.repeat(np.arange(n_samples)[:, None], n_coefs, axis=1)
        cols = np.reshape(np.arange(len(p)), (n_samples, n_coefs))
        jac[rows, cols] = Dm.T
        return jac

    def _preserve_minimum(p):
        P = np.reshape(p, (n_samples, n_coefs))
        return np.array([
            find_minimum(P[i], Dxl[i], Dxu[i]) - min_val
            for i in range(n_samples)])

    constraints = [
        {
            'type': 'eq',
            'fun': _preserve_continuity,
            'jac': _continuity_jac
        },
        {
            'type': 'eq',
            'fun': _preserve_mean,
            'jac': _mean_jac
        },
        {
            'type': 'ineq',
            'fun': _preserve_minimum
        }
    ]

    res = optimize.minimize(
        lambda p, x, fso, y: np.sum((_fit_func(p, x, fso) - y)**2),
        p_guess, (xi, force_second_order, yi), method='SLSQP',
        constraints=constraints,
        options={'disp': False, 'ftol': 1e-6, 'maxiter': 100}
    )

    return np.reshape(res.x, (n_samples, n_coefs))


class MeanPreservingInterpolation(object):

    def __init__(self, xi, yi, min_val=None, periodic=False,
                 x_edges=None, cubic_window_size=9):
        """
        Mean-preserving interpolation of a 1-D function.

        `xi` and `yi` are arrays of values from which the interpolation
        splines function `y = f(x)` is created such that the average of
        `f(x)` throughout an interval around `xi` is `yi`. The __call__
        method in this class triggers the interpolation magic.

        WARNING: Calling `MeanPreservingInterpolation` with NaNs in input
        values results in undefined behavior.

        Parameters
        ----------
        xi : (N,) array_like, or datetime_like
            Location of the interpolation values
        yi : (N,) array_like
            Interpolation Values
        min_val : None or float, optional
            Sets a minimum value that the interpolated values must respect. It
            is intended to constraint the interpolated values within physical
            meaningful limits. For instance, precipitation rates must be
            positive, hence, min_val=0. Default is None which, in practice, is
            alike min_val = -np.inf.
        periodic : bool, optional
            If True, the interpolation assumes periodic conditions.
            Default is False.
        x_edges : None or (N+1,) array_like, optional
            Define the intervals throughout which the mean of the splines must
            match the interpolation values `yi`.
            Default is None, meaning that `x_edges` are assumed at the center
            between consecutive `xi` values, and extrapolated at the leftmost
            and rightmost splines.
        cubic_window_size : int, optional
            When a `min_val` violation is found, it specifies the number of
            splines that are relaxed around such violation.
            Default is 9, meaning that 4 splines are relaxed on both sides
            of the violating spline.

        Methods
        -------
        __call__

        Examples
        --------
        >>> import numpy as np
        >>> import pylab as pl
        >>> from mpsplines import MeanPreservingInterpolation as MPI
        >>> # true process
        >>> x = np.linspace(0., 2*np.pi, 150)
        >>> y = np.sin(x)
        >>> # process sampling with averages (upscaling)
        >>> xi = np.reshape(x, (10, 15)).mean(axis=1)
        >>> yi = np.reshape(y, (10, 15)).mean(axis=1)
        >>> # interpolation from samples (averages) into the original grid
        >>> mpi = MPI(xi, yi)
        >>> xnew = x
        >>> ynew = mpi(xnew)
        >>> l, = pl.plot(x, y, 'k.', ms=8)
        >>> l, = pl.plot(xi, yi, 'c.', ms=16)
        >>> l, = pl.plot(xnew, ynew, 'r.', ms=4)
        >>> pl.show()
        """

        self.xi = np.asarray(xi).reshape(-1)
        self.yi = np.asarray(yi).reshape(-1)

        if self.xi.size != self.yi.size:
            raise AssertionError('`xi` and `yi` must have same size')

        if isinstance(self.xi[0], datetime):
            self.xi = self.xi.astype('datetime64[ns]')

        if np.issubdtype(self.xi.dtype, np.datetime64):
            dt64_ns = 'datetime64[ns]'
            ns_per_day = 24 * 3600 * 1e9
            trunc_day = self.xi.astype('datetime64[D]')
            frac_day = self.xi.astype(dt64_ns) - trunc_day.astype(dt64_ns)
            self.xi = trunc_day.astype(np.float64) + \
                frac_day.astype(np.float64) / ns_per_day

        self.xi = self.xi.astype(np.float64)

        self.n_samples = len(self.xi)
        self.min_val = min_val

        order = 3
        n_coefs = order + 1

        if int(cubic_window_size) % 2 != 1:
            raise AssertionError('cubic_window_size must be odd')

        cubic_window_half_size = int((cubic_window_size-1) / 2)

        if x_edges is None:
            self.x_edges = (self.xi[:-1] + self.xi[1:]) / 2.
            lower_bound = self.xi[0] - (self.xi[1] - self.xi[0]) / 2.
            upper_bound = self.xi[-1] + (self.xi[-1] - self.xi[-2]) / 2.
            self.x_edges = np.r_[lower_bound, self.x_edges, upper_bound]
        else:
            self.x_edges = np.asarray(x_edges).reshape(-1)
            assert len(self.x_edges) == (len(self.xi) + 1)

        # first, solve the problem using 2nd-order splines without
        # imposing a minimum value to the interpolated values. Thus,
        # the first column of P (coefficients for the 3rd-order term
        # of the splines) is initialized with zeros
        P = np.zeros((self.n_samples, n_coefs))
        P[:, 1:] = second_order_unconstrained_interpolation(
            self.xi, self.yi, self.x_edges, periodic)

        if self.min_val is None:
            self.P = P

        else:

            Dxl = self.x_edges[:-1] - self.xi
            Dxu = self.x_edges[1:] - self.xi

            # replace each 2nd-order spline that takes values below
            # self.min_val, by a 3rd-order spline constrained to take values
            # above that minimum. To that aim, take a window of width
            # 2*half_width (see below) around the 2nd-order spline and find a
            # 3rd-order spline that is above self.min_val, constrained to be
            # continuous in the boundaries with adjacent splines

            third_order_splines = [
                i for i in range(1, self.n_samples-1)  # no sampling bounds
                if find_minimum(P[i], Dxl[i], Dxu[i]) < self.min_val
            ]

            logger.debug(f'n_samples: {self.n_samples}')

            half_width = cubic_window_half_size
            while third_order_splines:  # tos: third-order spline
                # define window [min_i, max_i], starting from the first tos
                tos0 = third_order_splines[0]
                min_i = max(0, tos0 - half_width)
                max_i = min(tos0 + half_width + 1, self.n_samples - 1)
                while [tos for tos in third_order_splines
                       if (max_i < tos < max_i + half_width)]:
                    max_i = min(max_i + half_width + 1, self.n_samples - 1)

                window_tos = [tos - min_i for tos in third_order_splines
                              if (min_i < tos < max_i)]

                # window splines that will remain as actual 2nd order
                force_second_order = [k for k in range(max_i - min_i)
                                      if k not in window_tos]

                txt_tos = ', '.join([f'{k}' for k in third_order_splines])
                logger.info(f'Minimum-value violations at: {txt_tos}')
                logger.info(f'Local relaxed spline within [{min_i}, {max_i}]')

                # compute the 3rd-order splines for this window
                P[min_i: max_i] = third_order_constrained_interpolation(
                    self.xi[min_i: max_i], self.yi[min_i: max_i],
                    self.x_edges[min_i: max_i+1], self.min_val,
                    P[min_i: max_i], force_second_order)

                # update third-order splines
                third_order_splines = [tos for tos in third_order_splines
                                       if tos >= max_i]

            for i in range(1, self.n_samples-1):
                local_minimum = find_minimum(P[i], Dxl[i], Dxu[i])
                if (self.min_val - local_minimum) > 1e-6:
                    logger.warning(
                        f'the spline at position {i} could not be constrained '
                        f'above {self.min_val}. Local minimum={local_minimum}'
                    )

            self.P = P

    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : array_like
            1-D array of points to evaluate the interpolant at.

        Returns
        -------
        y : array_like
            Interpolated values.
        """

        x_ = np.array(x, ndmin=1).reshape(-1)

        if isinstance(x_[0], datetime):
            x_ = x_.astype('datetime64[ns]')

        if np.issubdtype(x_.dtype, np.datetime64):
            ns = 'datetime64[ns]'
            ns_per_day = 24 * 3600 * 1e9
            trunc_day = x_.astype('datetime64[D]')
            frac_day = x_.astype(ns) - trunc_day.astype(ns)
            x_ = trunc_day.astype(np.float64) + \
                frac_day.astype(np.float64) / ns_per_day

        x_ = x_.astype(np.float64)

        y = np.empty_like(x_)
        ind = np.clip(np.digitize(x_, self.x_edges)-1, 0, self.n_samples-1)
        for i in range(self.n_samples):
            universe = ind == i
            dx = x_[universe] - self.xi[i]
            y[universe] = (
                self.P[i, 0]*(dx**3) + self.P[i, 1]*(dx**2) +
                self.P[i, 2]*dx + self.P[i, 3])

        if self.min_val is None:
            return y
        return np.maximum(self.min_val, y)


def get_day_of_year(times):
    jan_1st = times.astype('datetime64[Y]').astype('datetime64[ns]')
    deltas = times.astype('datetime64[ns]') - jan_1st
    doys = deltas.astype('timedelta64[D]') + np.timedelta64(1, 'D')
    return doys.astype(float)


def get_number_of_days_in_year(times):
    one_day = np.timedelta64(1, 'D')
    one_year = np.timedelta64(1, 'Y')
    dec_31st = times.astype('datetime64[Y]') + one_year - one_day
    return get_day_of_year(dec_31st)


class MeanPreservingMonthlyLTAInterpolation(object):

    def __init__(self, yi, min_val=None, day_of_month=15, cubic_window_size=9):
        """
        Mean-preserving interpolation of a 1-D function of long-term averages.

        `xi` and `yi` are arrays of values used to approximate some function
        `y = f(x)` such that the average of `f(x)` throughout an interval
        around `xi` is `yi`. `xi` represents time and the interpolation assumes
        periodic boundary conditions.

        This class returns a function whose __call__ method uses interpolation
        to find the values at new points.

        Calling `MeanPreservingInterpolation` with NaNs in input values
        results in undefined behavior.

        Parameters
        ----------
        yi : (12,) array_like
            Monthly long-term average (LTA) interpolating values
        min_val : None or real, optional
            Specifies a global minimum value for the interpolated curve. It is
            intended to constraint the interpolated values within physically
            possible limits. For instance, precipitation rates must be always
            positive. Default is None, which, in practice, is like assuming
            min_val = -np.inf.
        day_of_month : float or array_like with shape (12,)
            day of month which the interpolaing values yi refers to
        cubic_window_size : int, optional
            Specifies the number of splines that are relaxed when a violation
            of min_val is found. The violating spline is in the middle of the
            window. Default is 9, meaning that 4 splines to each side of the
            violating spline are relaxed.

        Methods
        -------
        __call__

        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> import numpy as np
        >>> import pylab as pl
        >>> from mpsplines import MeanPreservingMonthlyLTAInterpolation as MPI
        >>> xi = [datetime(2018, month, 15, 12) for month in range(1, 13)]
        >>> yi = np.array([0.3078, 0.3072, 0.3084, 0.3132, 0.3254, 0.3314,\
                           0.3298, 0.3204, 0.3106, 0.3118, 0.3119, 0.3082])
        >>> # interpolation from samples (averages) into the original grid
        >>> mpi = MPI(yi)
        >>> xnew = [datetime(2017, 1, 1, 12) + timedelta(j) \
                    for j in range(365*3)]
        >>> ynew = mpi(xnew)
        >>> l, = pl.plot(xi, yi, 'k.', ms=16)
        >>> l, = pl.plot(xnew, ynew, 'r.', ms=4)
        >>> pl.show()
        """

        self.yi = np.asarray(yi).reshape(-1)

        if self.yi.size != 12:
            raise AssertionError(f'expected size of yi 12, got {self.yi.size}')

        dom = day_of_month
        if np.isscalar(dom):
            dom = np.full((12,), day_of_month)

        if np.any((dom < 1) | (dom > 31)):
            raise AssertionError('day_of_month out of bounds')

        days = np.array(
            [f'2020-{m:02d}-{d:02d}' for m, d in zip(range(1, 13), dom)],
            dtype='datetime64[ns]')
        print(days)
        xi = get_day_of_year(days) / get_number_of_days_in_year(days)
        print(xi)

        if np.any((xi < 0.) | (xi > 1)):
            raise AssertionError('xi out of bounds')

        # def hms_to_days(hms):
        #     seconds = hms.second + hms.microsecond / 1e6
        #     return (hms.hour + (hms.minute + seconds / 60.) / 60.) / 24.

        # ti = np.array(xi, ndmin=1, dtype='datetime64[us]').astype(datetime)
        # doy = [t.toordinal() -datetime(t.year, 1, 1).toordinal() for t in ti]
        # xi = [d + hms_to_days(t) for d, t in zip(doy, ti)]
        # self.xi = np.array(xi, dtype=np.float64) / 365.

        self._mpi = MeanPreservingInterpolation(
            xi, self.yi, min_val=min_val, periodic=True,
            x_edges=None, cubic_window_size=cubic_window_size)

    def __call__(self, x):
        """
        Evaluate the interpolant

        Parameters
        ----------
        x : datetime_like
            Interpolation times

        Returns
        -------
        y : array_like
            Interpolated values
        """

        ti = np.array(x, ndmin=1, dtype='datetime64[ns]')
        xi = get_day_of_year(ti) / get_number_of_days_in_year(ti)
        return self._mpi(xi)
