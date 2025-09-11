"""Utilities for testing with shapely geometries."""

from functools import partial

import numpy as np

import shapely

__all__ = ["assert_geometries_equal"]


def _equals_exact_with_ndim(x, y, tolerance):
    dimension_equals = shapely.get_coordinate_dimension(
        x
    ) == shapely.get_coordinate_dimension(y)
    with np.errstate(invalid="ignore"):
        # Suppress 'invalid value encountered in equals_exact' with nan coordinates
        geometry_equals = shapely.equals_exact(x, y, tolerance=tolerance)
    return dimension_equals & geometry_equals


def _replace_nan(arr):
    return np.where(np.isnan(arr), 0.0, arr)


def _assert_nan_coords_same(x, y, tolerance, err_msg, verbose):
    x, y = np.broadcast_arrays(x, y)
    x_coords = shapely.get_coordinates(x, include_z=True)
    y_coords = shapely.get_coordinates(y, include_z=True)

    # Check the shapes (condition is copied from numpy test_array_equal)
    if x_coords.shape != y_coords.shape:
        return False

    # Check NaN positional equality
    x_id = np.isnan(x_coords)
    y_id = np.isnan(y_coords)
    if not (x_id == y_id).all():
        msg = build_err_msg(
            [x, y],
            err_msg + "\nx and y nan coordinate location mismatch:",
            verbose=verbose,
        )
        raise AssertionError(msg)

    # If this passed, replace NaN with a number to be able to use equals_exact
    x_no_nan = shapely.transform(x, _replace_nan, include_z=True)
    y_no_nan = shapely.transform(y, _replace_nan, include_z=True)

    return _equals_exact_with_ndim(x_no_nan, y_no_nan, tolerance=tolerance)


def _assert_none_same(x, y, err_msg, verbose):
    x_id = shapely.is_missing(x)
    y_id = shapely.is_missing(y)

    if not (x_id == y_id).all():
        msg = build_err_msg(
            [x, y],
            err_msg + "\nx and y None location mismatch:",
            verbose=verbose,
        )
        raise AssertionError(msg)

    # If there is a scalar, then here we know the array has the same
    # flag as it everywhere, so we should return the scalar flag.
    if x.ndim == 0:
        return bool(x_id)
    elif y.ndim == 0:
        return bool(y_id)
    else:
        return y_id


def assert_geometries_equal(
    x,
    y,
    tolerance=1e-7,
    equal_none=True,
    equal_nan=True,
    normalize=False,
    err_msg="",
    verbose=True,
):
    """Raise an AssertionError if two geometry array_like objects are not equal.

    Given two array_like objects, check that the shape is equal and all elements
    of these objects are equal. An exception is raised at shape mismatch or
    conflicting values. In contrast to the standard usage in shapely, no
    assertion is raised if both objects have NaNs/Nones in the same positions.

    Parameters
    ----------
    x, y : Geometry or array_like
        Geometry or geometries to compare.
    tolerance: float, default 1e-7
        The tolerance to use when comparing geometries.
    equal_none : bool, default True
        Whether to consider None elements equal to other None elements.
    equal_nan : bool, default True
        Whether to consider nan coordinates as equal to other nan coordinates.
    normalize : bool, default False
        Whether to normalize geometries prior to comparison.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if normalize:
        x = shapely.normalize(x)
        y = shapely.normalize(y)
    x = np.asarray(x)
    y = np.asarray(y)

    is_scalar = x.ndim == 0 or y.ndim == 0

    # Check the shapes (condition is copied from numpy test_array_equal)
    if not (is_scalar or x.shape == y.shape):
        msg = build_err_msg(
            [x, y],
            err_msg + f"\n(shapes {x.shape}, {y.shape} mismatch)",
            verbose=verbose,
        )
        raise AssertionError(msg)

    flagged = False
    if equal_none:
        flagged = _assert_none_same(x, y, err_msg, verbose)

    if not np.isscalar(flagged):
        x, y = x[~flagged], y[~flagged]
        # Only do the comparison if actual values are left
        if x.size == 0:
            return
    elif flagged:
        # no sense doing comparison if everything is flagged.
        return

    is_equal = _equals_exact_with_ndim(x, y, tolerance=tolerance)
    if is_scalar and not np.isscalar(is_equal):
        is_equal = bool(is_equal[0])

    if np.all(is_equal):
        return
    elif not equal_nan:
        msg = build_err_msg(
            [x, y],
            err_msg + f"\nNot equal to tolerance {tolerance:g}",
            verbose=verbose,
        )
        raise AssertionError(msg)

    # Optionally refine failing elements if NaN should be considered equal
    if not np.isscalar(is_equal):
        x, y = x[~is_equal], y[~is_equal]
        # Only do the NaN check if actual values are left
        if x.size == 0:
            return
    elif is_equal:
        # no sense in checking for NaN if everything is equal.
        return

    is_equal = _assert_nan_coords_same(x, y, tolerance, err_msg, verbose)
    if not np.all(is_equal):
        msg = build_err_msg(
            [x, y],
            err_msg + f"\nNot equal to tolerance {tolerance:g}",
            verbose=verbose,
        )
        raise AssertionError(msg)


## BELOW A COPY FROM numpy.testing._private.utils (numpy version 1.20.2)


def build_err_msg(
    arrays,
    err_msg,
    header="Geometries are not equal:",
    verbose=True,
    names=("x", "y"),
    precision=8,
):
    msg = ["\n" + header]
    if err_msg:
        if err_msg.find("\n") == -1 and len(err_msg) < 79 - len(header):
            msg = [msg[0] + " " + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):
            if isinstance(a, np.ndarray):
                # precision argument is only needed if the objects are ndarrays
                r_func = partial(np.array_repr, precision=precision)
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f"[repr failed for <{type(a).__name__}>: {exc}]"
            if r.count("\n") > 3:
                r = "\n".join(r.splitlines()[:3])
                r += "..."
            msg.append(f" {names[i]}: {r}")
    return "\n".join(msg)
