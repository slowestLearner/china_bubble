#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Stock Data Processing

Created on Sat Jun 28 2025

Estimated running time: 5 sec
"""

import sys
import os

# Defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, "code", "util"))
CSMAR_DIR = os.path.join(BASE_DIR, "data", "csmar")
RESULTS_DIR_OLD = os.path.join(BASE_DIR, "output", "results")
RESULTS_DIR = os.path.join(BASE_DIR, "output_j", "results")
TABLES_DIR = os.path.join(BASE_DIR, "output_j", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "output_j", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sys.path.append(UTIL_DIR)
from general_import import *
import general_functions as gf

pd.options.display.max_columns = 50
pd.options.display.max_rows = 200
np.set_printoptions(suppress=True)

# pd.set_option('display.float_format',lambda x : '%.4f' % x)

import matplotlib.pyplot as plt
from linearmodels import PanelOLS, PooledOLS
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import compare
from collections import OrderedDict
from scipy.stats import mstats
import scipy.stats as stats
from format_results import *

# === 1. Stock Characteristics ===

stock_character = pd.read_parquet(
    os.path.join(RESULTS_DIR_OLD, "stock_character/stock_character_20220827")
)

# time since IPO. Those with less than a year since IPO are listed as outside assets
stock_character["ipo_gap"] = pd.to_datetime(
    stock_character["Month_End"], format="%Y%m%d"
) - pd.to_datetime(stock_character["IPO_DATE"], format="%Y%m%d")
stock_character["ipo_gap"] = stock_character["ipo_gap"].dt.days
stock_character = stock_character[stock_character["ipo_gap"] > 365]

# change ipo_gap to the unit of years
stock_character["ipo_gap"] = stock_character["ipo_gap"] / 365

# choose sample from 2011 to 2019
stock_character = stock_character[
    stock_character["Month_End"].between(20110000, 20199999)
]

stock_character["SOE"] = stock_character.StockAttr.isin(
    ["地方国有企业", "中央国有企业"]
).astype("int")

# import me_iv - NOTE: this is Chen's instrument, right?
me_IV = pd.read_parquet(os.path.join(RESULTS_DIR_OLD, "stock_character/me_iv"))

# NOTE: this (TYPE2 == 'SNQR' is the instrument specification used in the paper, right?
me_IV = me_IV[me_IV.TYPE2 == "SNQR"]
me_IV = me_IV[["SEC_CODE", "me_iv_equal", "me_iv_weight", "Month_End"]]
me_IV.columns = ["SEC_CODE", "me_iv1", "me_iv2", "Month_End"]  # rename

# NOTE: these zero values in instruments just meant that these are not held by anyone, right?
me_IV["me_iv1"] = np.where(
    me_IV["me_iv1"] <= 1, 1, me_IV["me_iv1"]
)  # replace me_iv1 < 1 with 1
me_IV["me_iv1_log"] = np.log(me_IV["me_iv1"])
me_IV["me_iv1_log_square"] = me_IV["me_iv1_log"] * me_IV["me_iv1_log"]

me_IV["me_iv2"] = np.where(
    me_IV["me_iv2"] <= 1, 1, me_IV["me_iv2"]
)  # replace me_iv2 < 1 with 1
me_IV["me_iv2_log"] = np.log(me_IV["me_iv2"])
me_IV["me_iv2_log_square"] = me_IV["me_iv2_log"] * me_IV["me_iv2_log"]

stock_character = stock_character.merge(me_IV, on=["SEC_CODE", "Month_End"], how="left")
del me_IV

# === 2. Data Processing ===

# most of the past returns are not NA, but lost of D/B are
na_cols = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "ret_lag_6",
    "ret_lag_7",
    "ret_lag_8",
    "ret_lag_9",
    "ret_lag_10",
    "ret_lag_11",
    "dividend_be",
]

# fill missing values with 0
stock_character[na_cols] = stock_character[na_cols].fillna(0)

# these columns cannot have NAs. Deleted around 4% of entries
drop_cols = [
    "SEC_CODE",
    "TRADE_DATE",
    "INDUSTRYNAME",
    "Book_Value",
    "ln_be",
    "beta",
    "Asset_YoY",
    "Book_YoY",
    "Profitability",
    "past_cumret_12",
]

stock_character = stock_character.replace([np.inf, -np.inf], np.nan)
stock_character = stock_character.dropna(subset=drop_cols)

# rename date column to TRADE_DATE
stock_character["TRADE_DATE"] = stock_character["Month_End"]

stock_character["price"] = np.exp(stock_character["ln_price"])

# import stock beta before 2014
stock_ret_beta = pd.read_parquet(
    os.path.join(RESULTS_DIR_OLD, "stock_character/stock_beta_before2014")
)
stock_character = stock_character.merge(
    stock_ret_beta[["SEC_CODE", "beta_b2014"]], how="left"
)

# import monthly stock returns
mkt_ret = pd.read_csv(os.path.join(CSMAR_DIR, "stocks_info/mkt_ret_month_05_20.csv"))

# renmaming variables
mkt_ret.rename(columns={"Trdmnt": "TRADE_DATE"}, inplace=True)
mkt_ret.rename(columns={"Mretmdtl": "mkt_ret"}, inplace=True)

# Append market return. keep only market returns calculated from stocks listed in SSE
mkt_ret = mkt_ret[mkt_ret["Markettype"] == 1]
mkt_ret = gf.convert_date(mkt_ret, "TRADE_DATE", "TRADE_DATE")
stock_character = (
    stock_character.assign(
        TRADE_MONTH=stock_character["TRADE_DATE"].astype(str).str[:6]
    )
    .merge(
        mkt_ret.assign(TRADE_MONTH=mkt_ret["TRADE_DATE"].astype(str).str[:6])[
            ["TRADE_MONTH", "mkt_ret"]
        ],
        on="TRADE_MONTH",
        how="left",
    )
    .drop(columns="TRADE_MONTH")
)

# import monthly risk-free rate
with open(os.path.join(CSMAR_DIR, "stocks_info/TRD_Nrrate.csv"), encoding="utf-8") as _:
    risk_free = pd.read_csv(_, parse_dates=["Clsdt"])

risk_free["TRADE_DATE"] = risk_free["Clsdt"].astype("str")
risk_free = gf.convert_date(risk_free, "TRADE_DATE", "TRADE_DATE")
risk_free = risk_free[["TRADE_DATE", "Nrrmtdt"]]
risk_free.columns = ["TRADE_DATE", "Rf"]
risk_free["Rf"] = risk_free["Rf"] / 100
stock_character = pd.merge(stock_character, risk_free, on=["TRADE_DATE"], how="left")

# compute beta-adjusted return (especially CAPM residual)
stock_character["beta_adj_ret"] = (
    stock_character["ret"] - stock_character["Rf"]
) - stock_character["beta_b2014"] * (stock_character["mkt_ret"] - stock_character["Rf"])

### Save Processed Data ###
to_dir = RESULTS_DIR + "/stock_character/"
os.makedirs(to_dir, exist_ok=True)
stock_character.to_parquet(
    # os.path.join(RESULTS_DIR, "stock_character/stock_processed"), index=False
    os.path.join(to_dir, "stock_processed.parquet"),
    index=False,
)

# # = checked
# tmp = pd.read_parquet('../../output/results/stock_character/stock_processed')
# stock_character.equals(tmp)

# === 3. Summary Statistics ===


# Industry Dummy
stock_character = pd.get_dummies(
    stock_character, columns=["INDUSTRYNAME"], drop_first=False, dtype=np.float64
)

# Change time variable into TRADE_DATE
stock_character["TRADE_DATE"] = stock_character["Month_End"]


# NOTE: this winsorization is used often, consider making a utility function. I also recall there are some built-in methods for it?
# Winsorize variables in the cross-section
def winsor(target, left, right):
    result = target.copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    mask = result.notna()
    result[mask] = mstats.winsorize(result[mask], limits=[left, right])
    return result


# winsorize by 1% on each side by month
def winsor_month_data(month_data, var_list):
    for var in var_list:
        month_data[var + "_win"] = winsor(month_data[var], 0.01, 0.01)
    return month_data


# variables to winsorize
winsor_list = ["Profitability", "Asset_YoY", "Book_YoY", "dividend_be"]

stock_character = stock_character.groupby(["TRADE_DATE"]).apply(
    lambda month: winsor_month_data(month, winsor_list)
)

# Drop TRADE_DATE from the index to avoid duplication
if "TRADE_DATE" in stock_character.index.names:
    stock_character.index = stock_character.index.droplevel("TRADE_DATE")

# Get cross-sectional z-scores
zscore_cols = [
    "past_cumret_12",
    "beta",
    "ln_be",
    "Profitability_win",
    "Asset_YoY_win",
    "Book_YoY_win",
    "dividend_be_win",
    "ipo_gap",
]

zscore_cols_name = [x + "_z" for x in zscore_cols]

stock_character[zscore_cols_name] = stock_character.groupby("TRADE_DATE")[
    zscore_cols
].transform(lambda x: stats.zscore(x, nan_policy="omit"))

temp = stock_character.copy()

var_list = [
    "Market_to_Book",
    "beta",
    "Book_Value",
    "Profitability_win",
    "Book_YoY_win",
    "dividend_be_win",
    "ipo_gap",
    "SOE",
    "past_cumret_12",
]

temp["Market_to_Book"] = stock_character["total_value"] / stock_character["Book_Value"]
temp["Book_Value"] = stock_character["Book_Value"] / 1e9
temp["past_cumret_12"] = stock_character["past_cumret_12"] * 100

# create summary by time periods
early = temp[temp.TRADE_DATE.between(20110000, 20139999)]
middle = temp[temp.TRADE_DATE.between(20140000, 20169999)]
late = temp[temp.TRADE_DATE.between(20170000, 20199999)]

# NOTE: these "%" are not escaped in latex. Not needed now, but later need to add escapes, etc.
early_summary = early.groupby("SEC_CODE")[var_list].mean()
early_summary = early_summary.describe().T[["count", "mean", "50%", "std"]]
early_summary.columns = ["count_early", "mean_early", "50%_early", "std_early"]

middle_summary = middle.groupby("SEC_CODE")[var_list].mean()
middle_summary = middle_summary.describe().T[["count", "mean", "50%", "std"]]
middle_summary.columns = ["count_middle", "mean_middle", "50%_middle", "std_middle"]

late_summary = late.groupby("SEC_CODE")[var_list].mean()
late_summary = late_summary.describe().T[["count", "mean", "50%", "std"]]
late_summary.columns = ["count_late", "mean_late", "50%_late", "std_late"]

summary = pd.concat([early_summary, middle_summary, late_summary], axis=1)

with open(os.path.join(TABLES_DIR, "summary_stat_stock_characteristics.tex"), "w") as f:
    f.write(summary.to_latex(float_format="%.2f"))

### Save Processed Data ###
stock_character.to_parquet(
    os.path.join(RESULTS_DIR, "stock_character/stock_processed_win.parquet"),
    index=False,
)

# # check, identical
# tmp = pd.read_parquet("../../output/results/stock_character/stock_processed_win")
# pd.testing.assert_frame_equal(
#     stock_character.reset_index(drop=True),
#     tmp.reset_index(drop=True),
#     check_dtype=False,
# )
