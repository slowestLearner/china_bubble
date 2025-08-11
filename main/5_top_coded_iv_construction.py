#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Top-Coded IV Construction

Created on Fri Jul 4 2025

Estimated running time: 37 min
"""

import sys
import os
import time
from joblib import Parallel, delayed

# defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, "code", "util"))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "output_j", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output_j", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "output_j", "tables")

RESULTS_DIR_OLD = os.path.join(BASE_DIR, "output", "results")

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
import statsmodels.sandbox.regression.gmm as gmm
import statsmodels.api as sm

from format_results import *

# load stock characteristics data
stock_character = pd.read_parquet(
    os.path.join(RESULTS_DIR, "stock_character/stock_processed_win.parquet")
)


# function to merge retail and institution files together
def combine_month_data(retail_file, institution_file, keep_cols, stock_character):
    # retail data
    retail_data = pd.read_parquet(retail_file, columns=keep_cols)
    retail_data = retail_data[retail_data["inside"] == 1]  # keep inside assets
    retail_data["investor_type"] = "retail"

    # institutional data
    institution_data = pd.read_parquet(institution_file, columns=keep_cols)
    institution_data = institution_data[
        institution_data["inside"] == 1
    ]  # keep inside assets
    institution_data["investor_type"] = "institution"

    # merge investors data
    month_data = pd.concat([retail_data, institution_data], ignore_index=True)

    # merge stock characteristics data
    month_data = pd.merge(
        month_data, stock_character, on=["SEC_CODE", "TRADE_DATE"], how="left"
    )
    # drop rows where 'w_i_n_adjust' is np.inf
    # month_data = month_data[month_data['w_i_n_adjust'] < np.inf]
    # drop groups with observations less than 150
    month_data["obs_n"] = month_data.groupby(["group_label"])["SEC_CODE"].transform(
        "count"
    )
    # month_data = month_data[month_data['obs_n'] >= 150]

    return month_data


# first stage regression (instrumenting log market value)
def group_first_regression(group, controls, trade_date, industry_cols):
    group = group.copy()

    group_label = group["group_label"].iloc[0]
    group_type = group["investor_type"].iloc[0]
    #     if group.shape[0] <  150:
    #         print('{}_{} data is empty or with too few observations'.format(group_label,trade_date ))
    #         return None
    endog = group["ln_mkv"]
    temp = group[industry_cols].any().reset_index()
    temp.columns = ["industry_name", "if_exist"]
    missing_industry = temp[temp.if_exist == False]["industry_name"].values
    regre_cols = controls
    if len(missing_industry) > 0:
        regre_cols = [x for x in controls if x not in missing_industry]

    exog = group[
        regre_cols
        + ["me_iv1_log", "me_iv1_log_square", "me_iv2_log", "me_iv2_log_square"]
    ]
    exog = sm.add_constant(exog)

    res = sm.OLS(endog, exog).fit()
    hypotheses = "(me_iv1_log = 0), (me_iv1_log_square = 0),(me_iv2_log = 0), (me_iv2_log_square = 0)"
    F_val = res.f_test(hypotheses).fvalue

    res_df = res.params.reset_index()
    res_df.columns = ["var_name", "coef"]
    res_df["t_values"] = res.tvalues.values
    res_df["p_values"] = res.pvalues.values
    res_df["F_stats"] = F_val

    if len(missing_industry) > 0:
        append_df = {
            "var_name": missing_industry,
            "coef": 0,
            "t_values": 0,
            "p_values": 0,
        }
        res_df = pd.concat([res_df, pd.DataFrame(append_df)], ignore_index=True)
        sort_col = (
            ["const"]
            + controls
            + ["me_iv1_log", "me_iv1_log_square", "me_iv2_log", "me_iv2_log_square"]
        )
        res_df = res_df.set_index("var_name").loc[sort_col].reset_index()
    res_df["R2_adj"] = res.rsquared_adj
    res_df["group_label"] = group_label
    res_df["acct_type"] = group_type
    res_df["nobs"] = res.nobs
    res_df["TRADE_DATE"] = trade_date

    return res_df


# second stage regression (estimating demand elasticity)
def group_second_regression(group, controls, trade_date, save_path, industry_cols):
    group = group.copy()
    group = group.sort_values(["SEC_CODE"])
    group_label = group["group_label"].iloc[0]
    group_type = group["investor_type"].iloc[0]

    #     if group.shape[0] < 150:
    #         print('{}_{} data is empty or with too few observations'.format(group_label,trade_date ))
    #         return None
    endog = np.log(group["w_i_n_adjust"])

    temp = group[industry_cols].any().reset_index()
    temp.columns = ["industry_name", "if_exist"]
    missing_industry = temp[temp.if_exist == False]["industry_name"].values
    regre_cols = controls
    if len(missing_industry) > 0:
        regre_cols = [x for x in controls if x not in missing_industry]

    exog = group[regre_cols + ["ln_mkv"]]
    exog = sm.add_constant(exog)
    instrument = group[
        regre_cols
        + ["me_iv1_log", "me_iv1_log_square", "me_iv2_log", "me_iv2_log_square"]
    ]
    instrument = sm.add_constant(instrument)

    top_coded = False
    res = gmm.IV2SLS(endog, exog, instrument=instrument).fit()
    if res.params.values[-1] >= 1:
        top_coded = True
        endog = np.log(group["w_i_n_adjust"]) - 0.999 * group["ln_mkv"]
        exog = group[regre_cols]
        exog = sm.add_constant(exog)
        res = sm.OLS(endog, exog).fit()

    res_df = res.params.reset_index()
    res_df.columns = ["var_name", "coef"]
    res_df["t_values"] = res.tvalues.values
    res_df["p_values"] = res.pvalues.values

    if top_coded:
        append_df = {
            "var_name": ["ln_mkv"],
            "coef": 0.999,
            "t_values": 0,
            "p_values": 0,
        }
        sort_col = res_df["var_name"].to_list() + ["ln_mkv"]
        res_df = pd.concat([res_df, pd.DataFrame(append_df)], ignore_index=True)
        res_df = res_df.set_index("var_name").loc[sort_col].reset_index()

    if len(missing_industry) > 0:
        append_df = {
            "var_name": missing_industry,
            "coef": 0,
            "t_values": 0,
            "p_values": 0,
        }
        res_df = pd.concat([res_df, pd.DataFrame(append_df)], ignore_index=True)
        sort_col = ["const"] + controls + ["ln_mkv"]
        res_df = res_df.set_index("var_name").loc[sort_col].reset_index()
    res_df["R2_adj"] = res.rsquared_adj
    res_df["group_label"] = group_label
    res_df["acct_type"] = group_type
    res_df["nobs"] = res.nobs
    res_df["TRADE_DATE"] = trade_date

    resid = res.resid.values
    beta_params = res_df["coef"].values

    ols_results = dict()
    ols_results["beta_params"] = beta_params
    ols_results["epsilon"] = resid
    ols_results["epsilon_SEC_CODE"] = group.SEC_CODE.to_numpy()
    save_file = "regression_coef_{}_{}".format(group_label, trade_date)
    np.save(os.path.join(save_path, save_file), ols_results)
    return res_df


### Monthly Loops ###

# obtain time periods needed
mkt_ret = pd.read_csv(
    os.path.join(RESULTS_DIR_OLD, "stock_character/mkt_return_month.csv"),
    parse_dates=["TRADE_DATE"],
)
mkt_ret.TRADE_DATE = mkt_ret.TRADE_DATE.astype("str")
mkt_ret = gf.convert_date(mkt_ret, "TRADE_DATE", "TRADE_DATE")

trade_dates = pd.Series(sorted(mkt_ret.TRADE_DATE.unique()))
trade_dates = trade_dates[trade_dates.between(20110000, 20200000)].tolist()

# define columns used
industry_cols = [
    "INDUSTRYNAME_采掘",
    "INDUSTRYNAME_化工",
    "INDUSTRYNAME_钢铁",
    "INDUSTRYNAME_有色金属",
    "INDUSTRYNAME_电子",
    "INDUSTRYNAME_汽车",
    "INDUSTRYNAME_家用电器",
    "INDUSTRYNAME_食品饮料",
    "INDUSTRYNAME_纺织服装",
    "INDUSTRYNAME_轻工制造",
    "INDUSTRYNAME_医药生物",
    "INDUSTRYNAME_公用事业",
    "INDUSTRYNAME_交通运输",
    "INDUSTRYNAME_房地产",
    "INDUSTRYNAME_商业贸易",
    "INDUSTRYNAME_休闲服务",
    "INDUSTRYNAME_银行",
    "INDUSTRYNAME_非银金融",
    "INDUSTRYNAME_综合",
    "INDUSTRYNAME_建筑材料",
    "INDUSTRYNAME_建筑装饰",
    "INDUSTRYNAME_电气设备",
    "INDUSTRYNAME_机械设备",
    "INDUSTRYNAME_国防军工",
    "INDUSTRYNAME_计算机",
    "INDUSTRYNAME_传媒",
    "INDUSTRYNAME_通信",
]  # take 'INDUSTRYNAME_农林牧渔' as baseline

group_cols = [
    "SEC_CODE",
    "TRADE_DATE",
    "group_label",
    "inside",
    "HOLD_Value",
    "HOLD_BAL",
    "w_i_n_raw",
    "w_i_0",
    "w_i_n_adjust",
]

# NOTE: again, we can't have past returns as controls... if needed we can construct something like analyst revisions, but variables highly related to current price can't controlled for
exog_cols = [
    "past_cumret_12",
    "beta",
    "ln_be",
    "Profitability_win",
    "Book_YoY_win",
    "dividend_be_win",
    "ipo_gap",
    "SOE",
] + industry_cols

reg_version = "EqualandBook_topcode1"

second_path = os.path.join(RESULTS_DIR, "Regression_Results", reg_version, "IV2SLSBeta")
gf.create_path(second_path)

second_path_reg_summary = os.path.join(
    RESULTS_DIR, "Regression_Results", reg_version, "IV2SLSsummary"
)
gf.create_path(second_path_reg_summary)

first_path = os.path.join(RESULTS_DIR, "Regression_Results", reg_version, "IVTest")
gf.create_path(first_path)

# tt = time.time()
# for trade_date in tqdm(trade_dates[:30]):
#     print(trade_date)
#     retail_file = os.path.join(
#         RESULTS_DIR_OLD,
#         "group_regre_data_09062022",
#         "retail",
#         f"retail_group_{trade_date}",
#     )
#     insti_file = os.path.join(
#         RESULTS_DIR_OLD,
#         "group_regre_data_09062022",
#         "institution",
#         f"group_insti_{trade_date}",
#     )
#     month_data = combine_month_data(
#         retail_file, insti_file, group_cols, stock_character
#     )
#     group_list = sorted(month_data["group_label"].unique())
#     res_first = []
#     res_second = []
#     for group_label in group_list:
#         if group_label.split("_")[0] in [
#             "310",
#             "410",
#             "420",
#             "430",
#             "440",
#             "450",
#             "QFII",
#             "RQFII",
#             "annuity",
#             "social",
#         ]:
#             continue
#         group_regre = month_data[month_data.group_label == group_label]
#         res_df = group_first_regression(
#             group_regre, exog_cols, trade_date, industry_cols
#         )
#         res_first.append(res_df)
#         res_df = group_second_regression(
#             group_regre, exog_cols, trade_date, second_path, industry_cols
#         )
#         res_second.append(res_df)
#     res_first = pd.concat(res_first, ignore_index=True)
#     res_first.to_parquet(
#         "{}/reg_results_{}".format(first_path, trade_date), index=False
#     )
#     res_second = pd.concat(res_second, ignore_index=True)
#     res_second.to_parquet(
#         "{}/reg_results_{}".format(second_path_reg_summary, trade_date), index=False
#     )

# print(f"running sequentially took a total of {((time.time() - tt) / 60):.2f} mins")


# helper function to process one trade date
def process_one_trade_date(trade_date):
    retail_file = os.path.join(
        RESULTS_DIR_OLD,
        "group_regre_data_09062022",
        "retail",
        f"retail_group_{trade_date}",
    )
    insti_file = os.path.join(
        RESULTS_DIR_OLD,
        "group_regre_data_09062022",
        "institution",
        f"group_insti_{trade_date}",
    )
    month_data = combine_month_data(
        retail_file, insti_file, group_cols, stock_character
    )
    group_list = sorted(month_data["group_label"].unique())
    res_first = []
    res_second = []
    for group_label in group_list:
        if group_label.split("_")[0] in [
            "310",
            "410",
            "420",
            "430",
            "440",
            "450",
            "QFII",
            "RQFII",
            "annuity",
            "social",
        ]:
            continue
        group_regre = month_data[month_data.group_label == group_label]
        res_df = group_first_regression(
            group_regre, exog_cols, trade_date, industry_cols
        )
        res_first.append(res_df)
        res_df = group_second_regression(
            group_regre, exog_cols, trade_date, second_path, industry_cols
        )
        res_second.append(res_df)
    res_first = pd.concat(res_first, ignore_index=True)
    res_first.to_parquet(
        "{}/reg_results_{}".format(first_path, trade_date), index=False
    )
    res_second = pd.concat(res_second, ignore_index=True)
    res_second.to_parquet(
        "{}/reg_results_{}".format(second_path_reg_summary, trade_date), index=False
    )


# NOTE: I suggest running this in parallel.
#   Testing: Running 30 dates in sequential mode took 2.03 mins. Doing the same in parallel mode (6 cores, a 5-year-old map laptop) took 0.71 mins
#   I ended up running the whole thing in parallel. The whole thing took 12.6 mins
tt = time.time()
Parallel(n_jobs=os.cpu_count() - 2, backend="loky", batch_size=1)(
    delayed(process_one_trade_date)(trade_date) for trade_date in trade_dates
)
print(f"parallel processing took {((time.time() - tt) / 60):.2f} mins")

# # NOTE: I get these error messages. Seems like there are some NAN in the data that goes into the regressions... consider debugging later:
# /Users/slowlearner/Dropbox/SpeculativeIdeas/china_demand/4_Code/bubble_data_and_code_reorganized/code_j/.venv/lib/python3.13/site-packages/statsmodels/sandbox/regression/gmm.py:147: RuntimeWarning: invalid value encountered in dot
#   Fty = np.dot(F.T, y)
# /Users/slowlearner/Dropbox/SpeculativeIdeas/china_demand/4_Code/bubble_data_and_code_reorganized/code_j/.venv/lib/python3.13/site-packages/statsmodels/regression/linear_model.py:1743: RuntimeWarning: invalid value encountered in subtract
#   centered_endog = model.wendog - model.wendog.mean()

# # spot check: check that results are identical
# out_new = pd.read_parquet(
#     "../../output_j/results/Regression_Results/EqualandBook_topcode1/IV2SLSsummary/reg_results_20191231"
# )
# out_old = pd.read_parquet(
#     "../../output/results/Regression_Results/EqualandBook_topcode1/IV2SLSsummary/reg_results_20191231"
# )

# pd.testing.assert_frame_equal(
#     out_new.reset_index(drop=True),
#     out_old.reset_index(drop=True),
#     check_dtype=False,
# )


# out_new = pd.read_parquet(
#     "../../output_j/results/Regression_Results/EqualandBook_topcode1/IVTest/reg_results_20190131"
# )
# out_old = pd.read_parquet(
#     "../../output/results/Regression_Results/EqualandBook_topcode1/IVTest/reg_results_20190131"
# )

# pd.testing.assert_frame_equal(
#     out_new.reset_index(drop=True),
#     out_old.reset_index(drop=True),
#     check_dtype=False,
# )
