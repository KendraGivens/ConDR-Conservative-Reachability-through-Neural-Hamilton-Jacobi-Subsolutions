"""Single source of truth for evaluation grids.

`hj_reachability` builds a grid's coordinate vectors as

    linspace(lo, hi, n, endpoint = bc is not periodic)

(`hj_reachability/grid.py`). So a PERIODIC dimension -- every heading angle in
this repo -- has **no node at the upper bound** and spacing `(hi-lo)/n`, while a
non-periodic dimension includes both endpoints and has spacing `(hi-lo)/(n-1)`.

Sampling a model on `linspace(..., endpoint=True)` and comparing it against a
periodic ground-truth axis offsets the comparison by up to a full cell. That bug
was fixed once in `eval_air3d.py` and left in nine other places, including the
one that evaluates the DeepReach baseline (audit P1-1). Use these helpers
everywhere instead of hand-rolling `linspace`.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def linear_axis(bounds, n):
    """Axis for a NON-periodic dimension (x, y, z, velocities, ...)."""
    lo, hi = bounds
    return np.linspace(float(lo), float(hi), int(n))


def periodic_axis(bounds, n):
    """Axis for a PERIODIC dimension (headings). No node at `hi`."""
    lo, hi = bounds
    return np.linspace(float(lo), float(hi), int(n), endpoint=False)


# Readable alias: every periodic dimension in this repo is an angle.
angle_axis = periodic_axis


def nearest_index(axis, values):
    """Index of the NEAREST node.

    `np.searchsorted` returns the upper insertion point, not the nearest node,
    which biases every lookup by up to one cell (audit P1-6).
    """
    axis = np.asarray(axis)
    return np.abs(np.asarray(values)[..., None] - axis).argmin(axis=-1)


def periodic_nearest_index(axis, values, period=TWO_PI):
    """Nearest node on a periodic axis, wrapping across the seam."""
    axis = np.asarray(axis)
    v = np.asarray(values)
    d = np.abs(((v[..., None] - axis) + period / 2.0) % period - period / 2.0)
    return d.argmin(axis=-1)


def wrap_to(values, lo=-np.pi, period=TWO_PI):
    """Wrap angles into [lo, lo + period)."""
    return ((np.asarray(values) - lo) % period) + lo


def periodic_interpolator(axes, values, periodic_dim, period=TWO_PI, fill_value=np.inf):
    """`RegularGridInterpolator` over a grid with one periodic dimension.

    Appends a wrapped copy of the first periodic slice at `lo + period` so the
    interpolator can cross the seam instead of clamping or returning `fill_value`
    there. `axes` must be built with `linear_axis` / `periodic_axis`.

    Returns a callable taking an (N, ndim) array of points.
    """
    from scipy.interpolate import RegularGridInterpolator

    axes = [np.asarray(a) for a in axes]
    pd = periodic_dim % len(axes)

    ax = axes[pd]
    ext_axis = np.concatenate([ax, [ax[0] + period]])
    first = np.take(values, [0], axis=pd)
    ext_values = np.concatenate([values, first], axis=pd)

    ext_axes = list(axes)
    ext_axes[pd] = ext_axis
    interp = RegularGridInterpolator(
        tuple(ext_axes), ext_values, bounds_error=False, fill_value=fill_value
    )

    lo = float(ax[0])

    def _call(points):
        pts = np.array(points, dtype=float, copy=True)
        pts[..., pd] = wrap_to(pts[..., pd], lo=lo, period=period)
        return interp(pts)

    return _call
