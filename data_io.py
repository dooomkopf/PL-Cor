#!/usr/bin/env python3
"""Data loading for the Bitcoin scaling-law analysis (PL-Cor)."""
import csv
import numpy as np
from datetime import date as _date

GENESIS = _date(2009, 1, 3)          # Bitcoin genesis block


def _to_float(s):
    """Parse a CSV cell to float; empty or invalid -> NaN."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def load_cm(path):
    """Load CoinMetrics daily CSV 'date,P,N,H' into a dict of arrays.

    Returns {'date': str[], 'P': float[], 'N': float[], 'H': float[]};
    missing values become NaN. Row filtering is left to the caller.
      P = PriceUSD, N = AdrActCnt (active addresses), H = HashRate [TH/s]
    """
    date, P, N, H = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            date.append(row['date'])
            P.append(_to_float(row['P']))
            N.append(_to_float(row['N']))
            H.append(_to_float(row['H']))
    return {'date': np.array(date),
            'P': np.array(P), 'N': np.array(N), 'H': np.array(H)}


def days_since_genesis(date_strings):
    """Network age t [days since the genesis block] for each ISO date string."""
    return np.array([(_date.fromisoformat(s) - GENESIS).days
                     for s in date_strings], dtype=float)


def positive(*arrays):
    """Mask given arrays to rows where ALL entries are finite and > 0.

    Required before any log-log analysis (log undefined for <= 0).
    Returns the arrays in the same order, jointly masked.
    """
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a) & (a > 0)
    return [a[mask] for a in arrays]
