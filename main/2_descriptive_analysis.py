#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Descriptive Analysis

Created on Sat Jun 28 2025

Estimated running time: 5 sec
"""

import sys
import os

# Defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, "code", "util"))
CSMAR_DIR = os.path.join(BASE_DIR, "data", "csmar")
RESULTS_DIR = os.path.join(BASE_DIR, "output_j", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output_j", "figures")

sys.path.append(UTIL_DIR)
from general_import import *
import general_functions as gf

pd.options.display.max_columns = 50
pd.options.display.max_rows = 200
np.set_printoptions(suppress=True)

# pd.set_option('display.float_format',lambda x : '%.4f' % x)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from linearmodels import PanelOLS, PooledOLS
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import compare
from collections import OrderedDict
from scipy.stats import mstats
import scipy.stats as stats

from format_results import *

# === 1. SSE Composite Index Overview ===
mkt_ret = pd.read_csv(
    os.path.join(RESULTS_DIR, "stock_character/market_return_daily.csv")
)
mkt_ret["TRADE_DATE"] = pd.to_datetime(
    mkt_ret["TRADE_DATE"], infer_datetime_format=True
)
mkt_ret["TRADE_DATE"] = mkt_ret["TRADE_DATE"].dt.strftime("%Y%m%d").astype("int")
mkt_ret = mkt_ret[mkt_ret.TRADE_DATE.between(20110000, 20200000)]
# mkt_ret = mkt_ret[mkt_ret.TRADE_DATE.between(20140000, 20170000)]
mkt_ret = mkt_ret.rename(columns={"index": "price_index"})

mkt_ret["cum_ret_market"] = ((mkt_ret.mkt_ret + 1).cumprod() - 1) * 100

### Full Sample ###
fig, ax = plt.subplots(figsize=(10, 6))
data = mkt_ret.copy()
data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"], format="%Y%m%d")
ax.plot(data.TRADE_DATE, data.price_index)
ax.set_ylabel("Price Index", fontsize=15)
ax.set_xlabel("Time", fontsize=15)
ax.set_title(f"The Time Series of Shanghai Stock Exchange Composite Index", fontsize=20)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_ylim([0, 5500])
# ax.set_ylim([7, 9])
plt.axvline(x=pd.to_datetime(20140701, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20141231, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20150612, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20160128, format="%Y%m%d"), color="k", linestyle="--")
fig.savefig(
    os.path.join(FIGURES_DIR, "price_index_full_sample.png"),
    dpi=300,
    bbox_inches="tight",
)

### Bubble Periods ###
fig, ax = plt.subplots(figsize=(10, 6))
data = mkt_ret.copy()
data["TRADE_DATE"] = pd.to_datetime(data["TRADE_DATE"], format="%Y%m%d")
ax.plot(data.TRADE_DATE, data.price_index)
ax.set_ylabel("Price Index", fontsize=15)
ax.set_xlabel("Time", fontsize=15)
# ax.set_title(f'The Time Series of Shanghai Stock Exchange Composite Index', fontsize = 20)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_ylim([0, 5500])
# ax.set_ylim([7, 9])
plt.axvline(x=pd.to_datetime(20140701, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20150101, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20150612, format="%Y%m%d"), color="k", linestyle="--")
plt.axvline(x=pd.to_datetime(20160131, format="%Y%m%d"), color="k", linestyle="--")
ax.set_xlim([datetime.date(2014, 1, 1), datetime.date(2016, 5, 30)])
ax.text(datetime.date(2014, 8, 1), 5000, "Formation", fontsize=15)
ax.text(datetime.date(2015, 1, 20), 5000, "Expansion", fontsize=15)
ax.text(datetime.date(2015, 8, 15), 5000, "Deflation", fontsize=15)
xtick = [
    datetime.date(2014, 7, 1),
    datetime.date(2015, 1, 1),
    datetime.date(2015, 6, 12),
    datetime.date(2016, 1, 31),
]  # datetime.date(2014, 1, 1),
ax.xaxis.set_ticks(xtick)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
fig.savefig(
    os.path.join(FIGURES_DIR, "price_index_bubble_period.png"),
    dpi=300,
    bbox_inches="tight",
)

# === 2. Illustrative Individual Stock Returns Examples ===

#### Raw Returns Calculation ###

price_daily = pd.read_parquet(
    os.path.join(RESULTS_DIR, "stock_character/stock_price_daily_20211130")
)

price_daily = price_daily[price_daily.TRADE_DATE.between(20140700, 20160200)]
price_daily = price_daily.sort_values(by=["SEC_CODE", "TRADE_DATE"])
price_daily["cum_ret"] = (
    price_daily.groupby(["SEC_CODE"])["ret"].transform(lambda x: (x + 1).cumprod() - 1)
    * 100
)

#### Abnornal Returns Calculation ###

# import stock beta before 2014 (computed in Step 1 file)
stock_ret_beta = pd.read_parquet(
    os.path.join(RESULTS_DIR, "stock_character/stock_beta_before2014")
)

price_daily = price_daily.merge(stock_ret_beta[["SEC_CODE", "beta_b2014"]], how="left")

# import market return data
mkt_ret = pd.read_csv(os.path.join(CSMAR_DIR, "stocks_info/mkt_return_daily.csv"))

# renmaming variables
mkt_ret.rename(columns={"Trddt": "TRADE_DATE"}, inplace=True)
mkt_ret.rename(columns={"Cdretmdtl": "mkt_ret"}, inplace=True)

# keep only market returns calculated from stocks in A-share maerets with ChiNext and STAR excluded
mkt_ret = mkt_ret[mkt_ret["Markettype"] == 5]

mkt_ret["TRADE_DATE"] = mkt_ret["TRADE_DATE"].astype("str")
mkt_ret = gf.convert_date(mkt_ret, "TRADE_DATE", "TRADE_DATE")

price_daily = price_daily.merge(
    mkt_ret[["TRADE_DATE", "mkt_ret"]], on=["TRADE_DATE"], how="left"
)

# import risk-free rate data
risk_free = pd.read_csv(
    os.path.join(CSMAR_DIR, "stocks_info/TRD_Nrrate.csv"), parse_dates=["Clsdt"]
)

risk_free["TRADE_DATE"] = risk_free["Clsdt"].astype("str")
risk_free = gf.convert_date(risk_free, "TRADE_DATE", "TRADE_DATE")
risk_free = risk_free[["TRADE_DATE", "Nrrdaydt"]]
risk_free.columns = ["TRADE_DATE", "Rf"]
risk_free["Rf"] = risk_free["Rf"] / 100

price_daily = pd.merge(price_daily, risk_free, on=["TRADE_DATE"], how="left")

# calculate beta adjusted return
price_daily["beta_adj_ret"] = (price_daily["ret"] - price_daily["Rf"]) - price_daily[
    "beta_b2014"
] * (price_daily["mkt_ret"] - price_daily["Rf"])

# calculate CAR (cumulative abnormal returns)
price_daily = price_daily.sort_values(by=["SEC_CODE", "TRADE_DATE"])
price_daily["cum_beta_adj_ret"] = (
    price_daily.groupby(["SEC_CODE"])["beta_adj_ret"].transform(
        lambda x: (x + 1).cumprod() - 1
    )
    * 100
)

### Plot Examples ###

key1 = "武钢股份(退市)"
# SEC_CODE = 600005
name1 = "Wuhan Iron and Steel Corporation"
data1 = price_daily[price_daily["SEC_Abbr"] == key1]
data1["TRADE_DATE"] = pd.to_datetime(data1["TRADE_DATE"], format="%Y%m%d")

key2 = "宝钢股份"
# SEC_CODE = 600019
name2 = "Baoshan Iron and Steel Corporation"
data2 = price_daily[price_daily["SEC_Abbr"] == key2]
data2["TRADE_DATE"] = pd.to_datetime(data2["TRADE_DATE"], format="%Y%m%d")

### Cumulative Returns ###
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data1.TRADE_DATE, data1.cum_beta_adj_ret, label=f"{name1}")
ax.plot(data2.TRADE_DATE, data2.cum_beta_adj_ret, label=f"{name2}")
plt.legend(fontsize=15)
ax.set_xlim([datetime.date(2014, 7, 1), datetime.date(2016, 1, 31)])
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_ylabel("Cumulative Abnormal Return (%)", fontsize=15)
# ax.set_title('Cumulative Abnormal Returns During the 2015 Bubble Period', fontsize=18)
fig.savefig(
    os.path.join(FIGURES_DIR, "cumulative_retrun_example.png"),
    dpi=300,
    bbox_inches="tight",
)

### Raw Retruns ###
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data1.TRADE_DATE, data1.cum_ret, label=f"{name1}")
ax.plot(data2.TRADE_DATE, data2.cum_ret, label=f"{name2}")
plt.legend(fontsize=15)
ax.set_xlim([datetime.date(2014, 7, 1), datetime.date(2016, 1, 31)])
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.set_ylabel("Raw Return (%)", fontsize=15)
# ax.set_title('Cumulative Raw Returns During the 2015 Bubble Period', fontsize=18)
fig.savefig(
    os.path.join(FIGURES_DIR, "raw_retrun_example.png"), dpi=300, bbox_inches="tight"
)
