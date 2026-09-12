"""Documented file contract for measured shot records (new in v1.12).

The tomography estimator (`estimate_squeezing`) and the fringe
contrast fit (`estimate_contrast`) both consume per-angle shot arrays
of measured J_z. This module gives that record a plain-text contract,
enforced instead of guessed at:

    angle_rad,jz

one row per shot; rows sharing an `angle_rad` value form that angle's
shot array, and angles keep their order of first appearance. The same
contract serves both records (tomography angles about the mean-spin
axis, or fringe phases of the final pulse), because they are the same
data shape. The save/load round trip is exact, asserted in the tests;
malformed files (wrong header, wrong field count, non-numeric values,
an angle with a single shot) are refused with an explanation. No
pandas: the standard library's csv module on top of NumPy.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_shots_csv", "save_shots_csv"]

_HEADER = ("angle_rad", "jz")


def save_shots_csv(path, angles, shots):
    """Write a shot record in the documented contract.

    angles : (K,) angles (radians). shots : length-K sequence of
    per-angle shot arrays (or a (K, M) array), J_z in spin units.
    """
    th = np.asarray(angles, dtype=float).ravel()
    rows = [np.asarray(s, dtype=float).ravel() for s in shots]
    if len(rows) != th.size or th.size == 0:
        raise ValueError("need one shot array per angle")
    if len(set(float(t) for t in th)) != th.size:
        raise ValueError("angles must be distinct (shots for one "
                         "angle belong in one array)")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER)
        for t, r in zip(th, rows):
            for v in r:
                w.writerow([repr(float(t)), repr(float(v))])


def load_shots_csv(path):
    """Read a shot record; returns (angles (K,), shots list of (M_k,)
    arrays), angles in order of first appearance -- ready to feed
    `estimate_squeezing` or `estimate_contrast`.

    Refusals instead of guesses: wrong header or field count,
    non-numeric values, or an angle carrying fewer than 2 shots (one
    shot has no variance and no standard error).
    """
    order = []
    groups = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = tuple(h.strip() for h in next(r))
        except StopIteration:
            raise ValueError("empty shot-record file") from None
        if header != _HEADER:
            raise ValueError(f"shot-record header must be exactly "
                             f"{_HEADER}; got {header}")
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != 2:
                raise ValueError(f"line {line}: expected 2 fields")
            t = float(rec[0])
            v = float(rec[1])
            if t not in groups:
                groups[t] = []
                order.append(t)
            groups[t].append(v)
    if not order:
        raise ValueError("shot-record file contains no data rows")
    for t in order:
        if len(groups[t]) < 2:
            raise ValueError(
                f"angle {t} carries {len(groups[t])} shot(s); at "
                "least 2 are needed for a variance or a standard "
                "error")
    return (np.asarray(order, dtype=float),
            [np.asarray(groups[t], dtype=float) for t in order])
