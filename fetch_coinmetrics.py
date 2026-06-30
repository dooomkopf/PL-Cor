#!/usr/bin/env python3
"""Fetch daily Bitcoin metrics from the CoinMetrics Community API into one CSV.

Columns written to cm_data.csv:  date, P (PriceUSD), N (AdrActCnt), H (HashRate).
"""
import os
import csv
import requests

# ------------------------------ parameters -------------------------------
HERE      = os.path.dirname(os.path.abspath(__file__))
OUT_CSV   = os.path.join(HERE, 'cm_data.csv')
API_URL   = 'https://community-api.coinmetrics.io/v4/timeseries/asset-metrics'
ASSET     = 'btc'
METRICS   = ['PriceUSD', 'AdrActCnt', 'HashRate']   # -> columns P, N, H
START     = '2009-01-01'        # earliest possible; API returns from first available
FREQ      = '1d'
PAGE_SIZE = 10000
# -------------------------------------------------------------------------

# CoinMetrics metric code -> short downstream column name.
COLMAP = {'PriceUSD': 'P', 'AdrActCnt': 'N', 'HashRate': 'H'}


def fetch():
    """Page through the API and collect {date: {P, N, H}}."""
    params = {'assets': ASSET, 'metrics': ','.join(METRICS),
              'frequency': FREQ, 'start_time': START, 'page_size': PAGE_SIZE}
    rows, url = {}, API_URL
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        for rec in j.get('data', []):
            day = rec['time'][:10]                 # 'YYYY-MM-DD'
            slot = rows.setdefault(day, {})
            for m in METRICS:
                if m in rec:
                    slot[COLMAP[m]] = rec[m]
        url = j.get('next_page_url')               # None when finished
        params = None                              # next_page_url is self-contained
    return rows


def main():
    rows = fetch()
    days = sorted(rows)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'P', 'N', 'H'])
        for day in days:
            s = rows[day]
            w.writerow([day, s.get('P', ''), s.get('N', ''), s.get('H', '')])
    print(f"{OUT_CSV}: {len(days)} rows, {days[0]} - {days[-1]}")


if __name__ == '__main__':
    main()
