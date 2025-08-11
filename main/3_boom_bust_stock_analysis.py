#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Boom Bust Stock Analysis

Created on Sat Jun 28 2025

Estimated running time: 10 sec
"""

import sys
import os

# Defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, "code", "util"))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "output_j", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output_j", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "output_j", "tables")


sys.path.append(UTIL_DIR)
from general_import import *
import general_functions as gf

pd.options.display.max_columns = 50
pd.options.display.max_rows = 200
np.set_printoptions(suppress=True)

# pd.set_option('display.float_format',lambda x : '%.4f' % x)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from linearmodels import PanelOLS, PooledOLS
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import compare
from collections import OrderedDict
from scipy.stats import mstats
from matplotlib.lines import Line2D

from format_results import *

# import data
stock_character = pd.read_parquet(
    os.path.join(RESULTS_DIR, "stock_character/stock_processed")
)

# === 1. Individual stock bubbles ===
# Identify individual bubbly stocks: Returns from 01/2014-03/2016

### Run-up ###
ret_201415 = stock_character[stock_character.TRADE_DATE.between(20140700, 20160200)]
ret_201415["raw_ret"] = ret_201415["ret"]
ret_201415["ret"] = ret_201415["beta_adj_ret"]
ret_201415 = ret_201415.dropna(subset=["ret"])
ret_201415 = ret_201415.sort_values(by=["SEC_CODE", "TRADE_DATE"])
ret_201415["cum_ret"] = (
    ret_201415.groupby(["SEC_CODE"])["ret"].transform(lambda x: (x + 1).cumprod() - 1)
    * 100
)

max_ret = ret_201415.loc[ret_201415.groupby(["SEC_CODE"])["cum_ret"].idxmax()][
    ["SEC_CODE", "TRADE_DATE", "cum_ret"]
]
max_ret.columns = ["SEC_CODE", "max_cum_ret_date", "max_cum_ret"]

kk = max_ret.groupby(["max_cum_ret_date"]).count()[["SEC_CODE"]]

### Crash ###
kk = ret_201415.merge(max_ret, on="SEC_CODE")
kk = kk[kk.TRADE_DATE >= kk.max_cum_ret_date]
bubble_size = kk.loc[kk.groupby(["SEC_CODE"])["cum_ret"].idxmin()][
    [
        "SEC_CODE",
        "TRADE_DATE",
        "cum_ret",
        "max_cum_ret_date",
        "max_cum_ret",
        "INDUSTRYCODE",
        "INDUSTRYNAME",
    ]
]
bubble_size.columns = [
    "SEC_CODE",
    "min_cum_ret_date",
    "min_cum_ret",
    "max_cum_ret_date",
    "max_cum_ret",
    "IndustryCode",
    "IndustryName",
]
kk = bubble_size.groupby(["min_cum_ret_date"]).count()[["SEC_CODE"]]

bubble_size["crash"] = (1 + bubble_size["min_cum_ret"] / 100) / (
    1 + bubble_size["max_cum_ret"] / 100
) - 1
bubble_size["crash"] = -bubble_size["crash"] * 100
bubble_size = bubble_size.rename(columns={"max_cum_ret": "runup"})
bubble_size = bubble_size.reindex(
    columns=[
        "SEC_CODE",
        "min_cum_ret_date",
        "min_cum_ret",
        "max_cum_ret_date",
        "runup",
        "crash",
        "IndustryCode",
        "IndustryName",
    ]
)

### Run-up and Crash ###
with open(os.path.join(TABLES_DIR, "summary_stat_boom_bust.tex"), "w") as f:
    f.write(bubble_size[["runup", "crash"]].describe().to_latex(float_format="%.2f"))

# === 2. Boom Bust V.S. Other Stock Attributes ===

# import retail share by stock
retail_hold_stock = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "additional_results_from_return_chasing_paper/retail_share/retail_institution_hold_by_stock.parquet",
    )
)

retail_hold_stock["retail_share"] = retail_hold_stock["Retail_Close_Value"] / (
    retail_hold_stock["Retail_Close_Value"]
    + retail_hold_stock["Institution_Close_Value"]
)

# import stock characteristics
characteristics = stock_character.copy()
stocks_MP_retail = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "additional_results_from_return_chasing_paper/MP_Decomposition/Stocks_MP",
    )
)
stocks_MP_retail = stocks_MP_retail[["SEC_CODE", "TRADE_DATE", "avg_MP_12"]]
stocks_MP_retail = stocks_MP_retail.rename(columns={"avg_MP_12": "MP_retail"})
characteristics = characteristics.merge(
    stocks_MP_retail, on=["SEC_CODE", "TRADE_DATE"], how="left"
)
characteristics = pd.merge(
    characteristics, retail_hold_stock, on=["SEC_CODE", "TRADE_DATE"], how="left"
)


# winsorize variables in the cross-section
def winsor(target, left, right):
    result = target.copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    mask = result.notna()
    result[mask] = mstats.winsorize(result[mask], limits=[left, right])
    return result


def winsor_month_data(month_data, var_list):
    for var in var_list:
        month_data[var + "_win"] = winsor(month_data[var], 0.01, 0.01)
    return month_data


winsor_list = [
    "PE",
    "PB",
    "Profitability",
    "Asset_YoY",
    "Book_YoY",
    "dividend_be",
    "ln_be",
]
characteristics = characteristics.groupby(["TRADE_DATE"]).apply(
    lambda month: winsor_month_data(month, winsor_list)
)

check_list = []
for x in winsor_list:
    check_list.append(x)
    check_list.append(x + "_win")

# get cross-sectional z-scores
zscore_cols = [
    "PE_win",
    "PB_win",
    "retail_share",
    "turnover",
    "MP_retail",
    "beta",
    "Book_YoY_win",
    "Asset_YoY_win",
    "ln_be",
    "ln_be_win",
    "Profitability_win",
    "dividend_be_win",
    "ipo_gap",
    "PE",
    "PB",
    "Book_YoY",
    "Asset_YoY",
    "Profitability",
    "dividend_be",
    "beta_b2014",
    "SOE",
    "past_cumret_12",
]

# drop TRADE_DATE from the index to avoid duplication
if "TRADE_DATE" in characteristics.index.names:
    characteristics.index = characteristics.index.droplevel("TRADE_DATE")

zscore_cols_name = [x + "_z" for x in zscore_cols]

characteristics[zscore_cols_name] = characteristics.groupby("TRADE_DATE")[
    zscore_cols
].transform(lambda x: stats.zscore(x, nan_policy="omit"))

### Identify big V.S. small bubbles ###

bubble_size["runup_sort"] = (
    pd.qcut(bubble_size["runup"], 2, labels=False, duplicates="drop") + 1
)
bubble_size["crash_sort"] = (
    pd.qcut(bubble_size["crash"], 2, labels=False, duplicates="drop") + 1
)

bubble_size["big_bubble"] = 0
bubble_size["small_bubble"] = 0
bubble_size.loc[
    (bubble_size["runup_sort"] == 2) & (bubble_size["crash_sort"] == 2), "big_bubble"
] = 1
bubble_size.loc[
    (bubble_size["runup_sort"] == 1) & (bubble_size["crash_sort"] == 1), "small_bubble"
] = 1

bubble_diff = characteristics.merge(bubble_size, on="SEC_CODE", how="inner")
bubble_diff = bubble_diff.sort_values(["SEC_CODE", "TRADE_DATE"])
bubble_diff["sector_cum_ret"] = (
    bubble_diff.groupby(["SEC_CODE"])["ind_ret"].transform(
        lambda x: (x + 1).cumprod() - 1
    )
    * 100
)
bubble_diff["ret_cum"] = (
    bubble_diff.groupby(["SEC_CODE"])["ret"].transform(lambda x: (x + 1).cumprod() - 1)
    * 100
)
bubble_diff["Month_End"] = pd.to_datetime(bubble_diff["Month_End"], format="%Y%m%d")
big_bubble = bubble_diff[bubble_diff["big_bubble"] == 1]
small_bubble = bubble_diff[bubble_diff["small_bubble"] == 1]
other_bubble = bubble_diff[bubble_diff["big_bubble"] == 0]

# pirnt big and small bubble stocks
print("Big Bubble Stocks:")
print(", ".join(big_bubble["SEC_CODE"].astype(str).unique()))
print("Small Bubble Stocks:")
print(", ".join(small_bubble["SEC_CODE"].astype(str).unique()))

# === 3. Character differences ===

### Druing bubble ###
temp = bubble_diff.copy()
temp["Month_End"] = pd.to_datetime(temp["Month_End"], errors="coerce")
temp = temp[
    (temp.Month_End >= pd.to_datetime(20140701, format="%Y%m%d"))
    & (temp.Month_End <= pd.to_datetime(20160131, format="%Y%m%d"))
]

example = temp[temp.SEC_CODE.apply(lambda x: x in set([600005, 600019]))]
example = example.groupby("SEC_Abbr")[
    ["retail_share", "beta_b2014", "Book_Value", "Profitability", "dividend_be"]
].mean()
example["Book_Value"] = example["Book_Value"] / 1e9
example[["retail_share", "Profitability", "dividend_be"]] = (
    example[["retail_share", "Profitability", "dividend_be"]] * 100
)
example = example[
    ["beta_b2014", "Book_Value", "retail_share", "Profitability", "dividend_be"]
]
with open(os.path.join(TABLES_DIR, "example.tex"), "w") as f:
    f.write(example.to_latex(float_format="%.2f"))

temp = temp.set_index(["SEC_CODE", "Month_End"])

dep_list = [
    "retail_share_z",
    "beta_z",
    "Book_YoY_z",
    "ln_be_z",
    "SOE_z",
    "Profitability_z",
    "dividend_be_z",
    "ipo_gap_z",
]
name_list = [
    "RetailShare",
    "CAPM Beta",
    "Investment",
    "Size",
    "SOE",
    "Profitability",
    "Dividend",
    "Age",
]

beta_list = []
err_list = []
for y in dep_list:
    mod = PanelOLS.from_formula(y + " ~ 1 + big_bubble + TimeEffects", temp)
    res = mod.fit(cov_type="robust")
    beta_list.append(res.params["big_bubble"])
    err_list.append(res.params["big_bubble"] - res.conf_int()["lower"]["big_bubble"])

coef_df = pd.DataFrame(
    list(zip(beta_list, err_list, dep_list)), columns=["coef", "err", "varname"]
)

# marker to use
marker_list = "s"
width = 0.25
color_list = ["tab:green"]
base_x = np.arange(len(dep_list))

### Pre-bubble and bubble ###
temp_prebubble = bubble_diff.copy()
temp_prebubble = temp_prebubble[
    (temp_prebubble.Month_End < pd.to_datetime(20140701, format="%Y%m%d"))
]
temp_prebubble = temp_prebubble.set_index(["SEC_CODE", "Month_End"])

beta_list = []
err_list = []
for y in dep_list:
    mod = PanelOLS.from_formula(y + " ~ 1 + big_bubble + TimeEffects", temp_prebubble)
    res = mod.fit(cov_type="robust")
    beta_list.append(res.params["big_bubble"])
    err_list.append(res.params["big_bubble"] - res.conf_int()["lower"]["big_bubble"])

coef_df_pre = pd.DataFrame(
    list(zip(beta_list, err_list, dep_list)), columns=["coef", "err", "varname"]
)

# marker to use
marker_list = "os"
width = 0.25
color_list = ["tab:purple", "tab:green"]
base_x = np.arange(len(dep_list)) - 0.2

### Visualize the results ###
fig, ax = plt.subplots(figsize=(18, 5))

# bar 1: pre-bubble
i = 0
X = base_x + width * i
ax.bar(
    X, coef_df_pre["coef"], yerr=coef_df_pre["err"], color="none", ecolor=color_list[i]
)
## axis labels
# ax.set_ylabel('$\hat{\gamma}_k$', fontsize=18)
plt.ylabel("$\hat{\gamma}_k$", rotation=0, fontsize=18)
ax.set_xlabel("Stock Characteristics", fontsize=18)
ax.scatter(
    x=X, marker=marker_list[i], s=120, y=coef_df_pre["coef"], color=color_list[i]
)
# bar 2: during-bubble
i = 1
X = base_x + width * i
ax.bar(X, coef_df["coef"], yerr=coef_df["err"], color="none", ecolor=color_list[i])
# ## remove axis labels
# ax.set_ylabel('$\hat{\gamma}_k')
# ax.set_xlabel('Stock Characteristics')
ax.scatter(x=X, marker=marker_list[i], s=120, y=coef_df["coef"], color=color_list[i])

ax.axhline(y=0, linestyle="--", color="black", linewidth=1)


ax.set_xticks(base_x + 0.1)
ax.set_xticklabels(name_list, rotation=0, fontsize=14)
ax.tick_params(axis="y", labelsize=15)
# ax.set_title(f'Preference for {name_list[seq]}', fontsize = 20)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker=marker_list[0],
        label="Pre-Bubble",
        color=color_list[0],
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker=marker_list[1],
        label="Bubble",
        color=color_list[1],
        markersize=10,
    ),
]
ax.legend(handles=legend_elements, loc=0, prop={"size": 12}, labelspacing=1.2)
fig.savefig(
    os.path.join(FIGURES_DIR, "difference_in_characteristics.png"),
    dpi=300,
    bbox_inches="tight",
)
