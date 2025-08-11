#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Demand Function Estimation

Created on Sun Jul 6 2025

Estimated running time:  30 sec
"""

import sys
import os

# defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, 'code',"util"))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "output", "tables")

sys.path.append(UTIL_DIR)
from general_import import *
from format_results import *
import general_functions as gf
pd.options.display.max_columns = 50
pd.options.display.max_rows =200
np.set_printoptions(suppress=True)

#pd.set_option('display.float_format',lambda x : '%.4f' % x)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from linearmodels import PanelOLS,PooledOLS
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import compare
from collections import OrderedDict
from scipy.stats import mstats
import scipy.stats as stats
import statsmodels.sandbox.regression.gmm as gmm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels import PanelOLS,PooledOLS
from copy import deepcopy
from statsmodels.iolib.table import SimpleTable
from matplotlib.lines import Line2D
import seaborn as sns

### Load Beat Coefficients ###

def get_month_res_dict(res_paths):
    res_dic ={}
    for _ in res_paths:
        res = np.load(_, allow_pickle=True).item()
        group_label = "_".join(_.split("/")[-1].split('_')[2:-1])
        res_dic[group_label] = res
    return res_dic

def get_params(res_dic):
    res_params = dict()
    for group_label in res_dic:
        res_params[group_label] = res_dic[group_label]['beta_params']
    return res_params

# get trade_dates list
with open(os.path.join(RESULTS_DIR, "stock_character", "mkt_return_month.csv"), 'r') as _:
    mkt_ret = pd.read_csv(_,parse_dates=['TRADE_DATE'])
mkt_ret.TRADE_DATE = mkt_ret.TRADE_DATE.astype("str")
mkt_ret = gf.convert_date(mkt_ret,"TRADE_DATE", "TRADE_DATE")
trade_dates = pd.Series(sorted(mkt_ret.TRADE_DATE.unique()))
trade_dates = trade_dates[trade_dates.between(20110000,20200000)].to_list()
trade_dates = [str(x) for x in trade_dates]

keep_cols = ['group_label','TRADE_DATE','HOLD_Value']

retail_hold = pd.read_parquet(os.path.join(RESULTS_DIR, "group_regre_data_09062022", "retail"), columns=keep_cols)
retail_hold = retail_hold.groupby(['TRADE_DATE','group_label'])['HOLD_Value'].sum().reset_index()
institute_hold = pd.read_parquet(os.path.join(RESULTS_DIR, "group_regre_data_09062022", "institution"), columns=keep_cols)
institute_hold = institute_hold.groupby(['TRADE_DATE','group_label'])['HOLD_Value'].sum().reset_index()
group_weight = pd.concat([retail_hold, institute_hold])

file_folder = glob.glob(os.path.join(RESULTS_DIR, "regression_results", "EqualandBook_topcode1", "IV2SLSBeta", "*"))

beta_raw = []
for trade_date in tqdm(trade_dates):
    res_paths = []
    for _ in file_folder:
        if _.split("/")[-1].split('_')[-1][:-4] == trade_date:
            res_paths.append(_)
    res_paths = sorted(res_paths)
    res_dic = get_month_res_dict(res_paths)
    res_params = get_params(res_dic)
    beta = pd.DataFrame.from_dict(res_params, orient='index').reset_index()
    beta = beta.rename(columns={"index": "group_label"})
    beta['TRADE_DATE'] = int(trade_date)
    beta_raw.append(beta)

select_cols = ['const', 'past_cumret_12', 'Beta','log_BE', 'Profitability_win', 'Book_YoY_win', 'dividend_be_win', 'Age','SOE', 'INDUSTRYNAME_采掘', 'INDUSTRYNAME_化工', 'INDUSTRYNAME_钢铁', 'INDUSTRYNAME_有色金属', 'INDUSTRYNAME_电子', 'INDUSTRYNAME_汽车', 'INDUSTRYNAME_家用电器', 'INDUSTRYNAME_食品饮料', 'INDUSTRYNAME_纺织服装', 'INDUSTRYNAME_轻工制造', 'INDUSTRYNAME_医药生物', 'INDUSTRYNAME_公用事业', 'INDUSTRYNAME_交通运输', 'INDUSTRYNAME_房地产', 'INDUSTRYNAME_商业贸易', 'INDUSTRYNAME_休闲服务', 'INDUSTRYNAME_银行', 'INDUSTRYNAME_非银金融', 'INDUSTRYNAME_综合', 'INDUSTRYNAME_建筑材料', 'INDUSTRYNAME_建筑装饰', 'INDUSTRYNAME_电气设备', 'INDUSTRYNAME_机械设备', 'INDUSTRYNAME_国防军工', 'INDUSTRYNAME_计算机', 'INDUSTRYNAME_传媒', 'INDUSTRYNAME_通信', 'log_price'] # take 'INDUSTRYNAME_农林牧渔' as the base industry

beta_all = pd.concat(beta_raw)
beta_all.columns = ['group_label',*select_cols,'TRADE_DATE']

beta_all = beta_all.merge(group_weight, on=['group_label','TRADE_DATE'],how='left')
beta_all = beta_all.sort_values(['TRADE_DATE','group_label'])
beta_all = beta_all[['group_label',*select_cols,'HOLD_Value','TRADE_DATE']]

beta_all['retail'] = (beta_all['group_label'].transform(lambda x: len(x.split("_"))) == 5).astype('int')

beta_bubble = beta_all.copy()

### Prep for Plots Later ###

beta_bubble['agg_type'] = beta_bubble['group_label'].transform(lambda x: x.split("_")[0])

beta_bubble['log_BE'] = beta_bubble['log_BE'] + beta_bubble['log_price'] # change the coefficient for log(BE) to log(BE) + log(ME): preference for size. coefficient for log(ME) is unchanged: variable correspondingly become log(ME) - log(BE)

# === 1. Demand Function Paremeters Summary Stats ===

temp = beta_bubble.copy()
temp['demand_elasticity'] = 1 - temp['log_price']

var_list = ['log_price','Beta', 'log_BE', 'Profitability_win', 'Book_YoY_win', 'dividend_be_win', 'Age', 'SOE','past_cumret_12']

early_retail = temp[(temp.TRADE_DATE.between(20110000, 20140000)) & (temp.retail == 1)]
early_insti = temp[(temp.TRADE_DATE.between(20110000, 20140000)) & (temp.retail ==0)]

mid_retail = temp[(temp.TRADE_DATE.between(20140000, 20170000)) & (temp.retail == 1)]
mid_insti = temp[(temp.TRADE_DATE.between(20140000, 20170000)) & (temp.retail ==0)]

late_retail = temp[(temp.TRADE_DATE.between(20170000, 20200000)) & (temp.retail == 1)]
late_insti = temp[(temp.TRADE_DATE.between(20170000, 20200000)) & (temp.retail ==0)]

col_list = ['mean','std'] #'50%',,'count',
name_list = ['early_retail', 'early_insti', 'mid_retail', 'mid_insti', 'late_retail', 'late_insti']
summary_list = []
for idx, df in enumerate([early_retail, early_insti, mid_retail, mid_insti, late_retail, late_insti]):
    kk = df.groupby('group_label')[var_list].mean()
    kk = kk.describe().T[col_list]
    kk.columns = [f'{x}_{name_list[idx]}' for x in col_list]
    summary_list.append(kk)

summary = pd.concat(summary_list, axis=1)
summary.index = ['demand_elasticity', 'Beta', 'log_book', 'Profitability','Investment', 'dividend_book', 'Age', 'SOE', 'past1yearRet']

with open(os.path.join(TABLES_DIR, "demand_function_parameters.tex"), "w") as f: f.write(summary.to_latex(float_format="%.2f"))

### Heterogeneous Preference ###

def winsor(target, left, right):
    result = target.copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    mask = result.notna()
    result[mask] = mstats.winsorize(result[mask], limits=[left, right])
    return result

def gen_value_weighted_coef(df):
    # df columns need to be: [coef. columns that need to be weighted, value, TRADE_DATE]
    df = df.copy()
    df.columns = [*df.columns[:-2],'value','TRADE_DATE']
    
    for col in df.columns[:-2]:
#         df[col] = df.groupby('TRADE_DATE')[col].transform(lambda x: winsor(x, 0.05, 0.05))
        df[col] = df[col]*df['value']
    weighted = df.groupby('TRADE_DATE')[df.columns[:-1]].sum()
    for col in df.columns[:-2]:
        weighted[col] = weighted[col]/weighted['value']
#     weighted.index = pd.to_datetime(weighted.index, format='%Y%m%d')
    return weighted

def gen_value_weighted_quantile(df, q = 0.75):
    # df columns need to be: [coef. columns that need to be weighted, value, TRADE_DATE]
    df = df.copy()
    df.columns = [*df.columns[:-2],'value','TRADE_DATE']
    df['share'] = df.groupby('TRADE_DATE')['value'].transform(lambda x: x/x.sum())
    rres = []
    for col in df.columns[:-3]:
        df = df.sort_values(['TRADE_DATE', col])
        df['cum_share'] = df.groupby('TRADE_DATE')['share'].cumsum()
        weighted_quantile_idx = df.groupby(['TRADE_DATE'])['cum_share'].agg(
            lambda x: (x-q).abs().idxmin())
        weighted_quantile = df.loc[weighted_quantile_idx][['TRADE_DATE',col]].set_index('TRADE_DATE')
        rres.append(weighted_quantile)
    rres = pd.concat(rres, axis=1)
    rres.index = pd.to_datetime(rres.index, format='%Y%m%d')
    return rres

def gen_equal_weighted_coef(df):
    # df columns need to be: [coef. columns that need to be weighted, value, TRADE_DATE]
    df = df.copy()
    df.columns = [*df.columns[:-2],'value','TRADE_DATE']
    
#     for col in df.columns[:-2]:
#         df[col] = df.groupby('TRADE_DATE')[col].transform(lambda x: winsor(x, 0.05, 0.05))
    equal_weighted = df.groupby('TRADE_DATE')[df.columns[:-1]].mean()
#     equal_weighted.index = pd.to_datetime(equal_weighted.index, format='%Y%m%d')
    return equal_weighted


# === 2. Retail v.s. Institution ===

### Time-series Mean ###
columns = beta_bubble.columns[1:-2]
beta_list_retail = gen_value_weighted_coef(beta_bubble[beta_bubble.retail==1][columns])
beta_list_institute = gen_value_weighted_coef(beta_bubble[beta_bubble.retail==0][columns])

beta_list_retail_q25 = gen_value_weighted_quantile(beta_bubble[beta_bubble.retail==1][columns], q=0.25)
beta_list_retail_q75 = gen_value_weighted_quantile(beta_bubble[beta_bubble.retail==1][columns], q=0.75)

beta_list_institute_q25 = gen_value_weighted_quantile(beta_bubble[beta_bubble.retail==0][columns], q=0.25)
beta_list_institute_q75 = gen_value_weighted_quantile(beta_bubble[beta_bubble.retail==0][columns], q=0.75)


beta_list_institute = gen_equal_weighted_coef(beta_bubble[beta_bubble.retail==0][columns])
beta_list_institute = beta_list_institute.reset_index()

### Regression on 5 Periods ###

Regression_Data = beta_bubble.copy()
Regression_Data['TRADE_DATE'] = Regression_Data['TRADE_DATE'].astype('int')

Regression_Data['period'] = ''
Regression_Data.loc[Regression_Data['TRADE_DATE']<20140700,'period'] = '1Pre Bubble'
Regression_Data.loc[(Regression_Data['TRADE_DATE']>20140700) & (Regression_Data['TRADE_DATE']<20141299),'period'] = '2Bubble Runup1'
Regression_Data.loc[(Regression_Data['TRADE_DATE']>20150000) & (Regression_Data['TRADE_DATE']<20150699),'period'] = '3Bubble Runup2'
Regression_Data.loc[(Regression_Data['TRADE_DATE']>20150700) & (Regression_Data['TRADE_DATE']<20160199),'period'] = '4Bubble Crash'
Regression_Data.loc[Regression_Data['TRADE_DATE']>20160200,'period'] = '5Post Bubble'

Regression_Data['type_period'] = ''
Regression_Data.loc[(Regression_Data.retail==0)&(Regression_Data.period=='1Pre Bubble'  ),'type_period'] = 'Institution_1Pre Bubble'  
Regression_Data.loc[(Regression_Data.retail==0)&(Regression_Data.period=='2Bubble Runup1'),'type_period'] = 'Institution_2Bubble Runup1'
Regression_Data.loc[(Regression_Data.retail==0)&(Regression_Data.period=='3Bubble Runup2'),'type_period'] = 'Institution_3Bubble Runup2'
Regression_Data.loc[(Regression_Data.retail==0)&(Regression_Data.period=='4Bubble Crash'),'type_period'] = 'Institution_4Bubble Crash'
Regression_Data.loc[(Regression_Data.retail==0)&(Regression_Data.period=='5Post Bubble' ),'type_period'] = 'Institution_5Post Bubble' 

Regression_Data.loc[(Regression_Data.retail==1)&(Regression_Data.period=='1Pre Bubble'  ),'type_period'] = 'Retail_1Pre Bubble'  
Regression_Data.loc[(Regression_Data.retail==1)&(Regression_Data.period=='2Bubble Runup1'),'type_period'] = 'Retail_2Bubble Runup1'
Regression_Data.loc[(Regression_Data.retail==1)&(Regression_Data.period=='3Bubble Runup2'),'type_period'] = 'Retail_3Bubble Runup2'
Regression_Data.loc[(Regression_Data.retail==1)&(Regression_Data.period=='4Bubble Crash'),'type_period'] = 'Retail_4Bubble Crash'
Regression_Data.loc[(Regression_Data.retail==1)&(Regression_Data.period=='5Post Bubble' ),'type_period'] = 'Retail_5Post Bubble'

Regression_Data = Regression_Data.set_index(['group_label','TRADE_DATE'])

Regression_Data['demand_elasticity'] = 1 - Regression_Data['log_price']

res_one_minus = []
var_list = ['log_price']
name_list = ['Log(Price)']
for y in var_list:
    Regression_Data['one_minus_' + y] = 1 - Regression_Data[y]
    mod = PanelOLS.from_formula(f'one_minus_{y} ~ type_period', Regression_Data)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    res_one_minus.append(res)

res_list = []
var_list = ['log_price', 'dividend_be_win','log_BE', 'Profitability_win','Beta','Age','INDUSTRYNAME_建筑装饰']
name_list = ['Log(Price)', 'Dividend','Log(Book Equity)','Profitability','CAPM Beta','Age','Construction Industry']
for y in var_list:
    mod = PanelOLS.from_formula(y + ' ~ type_period', Regression_Data) 
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    res_list.append(res)

### Save Regression Results ###

time_effects = ['N'] * 10
stock_effects = ['N'] * 10

result = compare_panel(res_list, stock_effects, time_effects, stars=True)
result_one_minus = compare_panel(res_one_minus, stock_effects, time_effects, stars=True)

rows_main = deepcopy(result.summary.tables[0].data)
rows_one_minus = deepcopy(result_one_minus.summary.tables[0].data)

cols_to_keep = [4, 5] # columns to keep from the result

combined_rows = []

for r1, r2 in zip(rows_one_minus, rows_main):
    row_label = r1[0] # use the left-most label
    one_minus_val = r1[1] # the first coefficient
    others = [r2[i] for i in cols_to_keep]
    combined_rows.append([row_label, one_minus_val] + others)

combined_table = SimpleTable(combined_rows)
latex_output = combined_table.as_latex_tabular()

# format Latex output as lccc but not cccc
n_cols = len(combined_rows[0])
default_align = 'c' * n_cols
custom_align = 'l' + 'c' * (n_cols - 1)
latex_output = latex_output.replace( f"\\begin{{tabular}}{{{default_align}}}", f"\\begin{{tabular}}{{{custom_align}}}")

with open(os.path.join(TABLES_DIR, "demand_function_coefficients_retail_institution.tex"), "w") as f: f.write(latex_output)

### Preferences/Beliefs Plotting ###

# marker to use
marker_list = 'so'
width=0.25
color_list = ['tab:orange', 'tab:blue']
# 5 periods in total
base_x = np.arange(5) - 0.25

for seq, res in enumerate(res_list[1:], start=1):
    err_series = res.params - res.conf_int()['lower']
    coef_df = pd.DataFrame({'coef': res.params, 'err': err_series, 'varname': err_series.index.str.slice(12, -1)})
    coef_df['invest_type'] = coef_df.varname.str.split(pat='_', expand=True)[0]
    coef_df['period'] = coef_df.varname.str.split(pat='_', expand=True)[1]
    coef_df = coef_df.reset_index(drop=True)
    coef_df = coef_df.drop(columns=['varname'])
    coef_df['var_name'] = var_list[seq]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    # single figure uses size (14,6); double figure uses size (12,6)
    for i, mod in enumerate(coef_df.invest_type.unique()):
        mod_df = coef_df[coef_df.invest_type == mod]
        mod_df = mod_df.set_index('period').reindex(coef_df['period'].unique())
        # offset x posistions
        X = base_x + width*i
        ax.bar(X, mod_df['coef'],yerr=mod_df['err'],color='none',ecolor=color_list[i])
        # remove axis labels
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.scatter(x=X, marker=marker_list[i], s=120, y=mod_df['coef'], color=color_list[i])
        ax.axhline(y=0, linestyle='--', color='black', linewidth=1)

        ax.set_xticks(base_x+0.1)
    #     ax.xaxis.set_ticks_position('default')
        ax.set_xticklabels(['Pre-Bubble','Formation','Expansion','Deflation','Post-Bubble'], rotation=0, fontsize=16)
        ax.tick_params(axis='y', labelsize = 15)
        
    # build customized legend
    legend_elements = [Line2D([0], [0], marker=marker_list[0], label=coef_df.invest_type.unique()[0], color = color_list[0], markersize=10), Line2D([0], [0], marker=marker_list[1], label=coef_df.invest_type.unique()[1], color = color_list[1],markersize=10)]
    ax.legend(handles=legend_elements, loc='best', prop={'size': 15}, labelspacing=1.2)
#     ax.legend(handles=legend_elements, loc=1, prop={'size': 15}, labelspacing=1.2)
    ax.set_title(f'Preferences/Beliefs about {name_list[seq]}', fontsize = 20)
#     ax.set_title(f'{name_list[seq]}', fontsize = 20)
    fig.savefig(os.path.join(FIGURES_DIR, f'demand_function_retail_institution_{name_list[seq]}.png'), dpi=300, bbox_inches='tight')

### Demand Elasiticity Plotting ###

seq = 0
res = res_list[seq]
err_series = res.params - res.conf_int()['lower']
coef_df = pd.DataFrame({'coef': 1 - res.params, 'err': err_series, 'varname': err_series.index.str.slice(12, -1)})
coef_df['invest_type'] = coef_df.varname.str.split(pat='_', expand=True)[0]
coef_df['period'] = coef_df.varname.str.split(pat='_', expand=True)[1]
coef_df = coef_df.reset_index(drop=True)
coef_df = coef_df.drop(columns=['varname'])
coef_df['var_name'] = var_list[seq]

fig, ax = plt.subplots(figsize=(14, 6))
for i, mod in enumerate(coef_df.invest_type.unique()):
    mod_df = coef_df[coef_df.invest_type == mod]
    mod_df = mod_df.set_index('period').reindex(coef_df['period'].unique())
    # offset x posistions
    X = base_x + width*i
    ax.bar(X, mod_df['coef'],yerr=mod_df['err'],color='none',ecolor=color_list[i])
    # remove axis labels
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=X, marker=marker_list[i], s=120, y=mod_df['coef'], color=color_list[i])
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)

    ax.set_xticks(base_x+0.1)
#     ax.xaxis.set_ticks_position('default')
    ax.set_xticklabels(['Pre-Bubble','Early Run-up','Late Run-up','Crash','Post-Bubble'], rotation=0, fontsize=16)
    ax.tick_params(axis='y', labelsize = 15)

# customize legend
legend_elements = [Line2D([0], [0], marker=marker_list[0], label=coef_df.invest_type.unique()[0], color = color_list[0], markersize=10), Line2D([0], [0], marker=marker_list[1], label=coef_df.invest_type.unique()[1], color = color_list[1], markersize=10)]
ax.legend(handles=legend_elements, loc=2, prop={'size': 12}, labelspacing=1.2)
ax.set_title(f'Demand Elasticity', fontsize = 20)
fig.savefig(os.path.join(FIGURES_DIR, f'demand_elasticity_retail_institution.png'), dpi=300, bbox_inches='tight')

# === 3. Untangling Retail Investors ===

# About group label:
# 1. First index: cohort – [ before, runnup1, runnup2, crash, post ]
# 2. Second index: stock market wealth – [ <100K, 100K-500K, 500K-5M, 5M-10M, >10M ] (CNY)
# 3. Third index: RCP level – split into 5 groups, where 1 = lowest MP
# 4. Fourth index: gender – 0 = male, 1 = female
# 5. Fifth index: age – divided into 3 groups with 1 = youngest

Regression_Data = beta_bubble[beta_bubble.retail == 1]
Regression_Data['RCP_type'] = Regression_Data['group_label'].transform(lambda x: x.split("_")[2]) # conditioning on RCP level
Regression_Data['cohort_type'] = Regression_Data['group_label'].transform(lambda x: x.split("_")[0]) # conditioning on cohort

Regression_Data.loc[Regression_Data.cohort_type == 'pre', 'cohort_type'] = '1pre'
Regression_Data.loc[Regression_Data.cohort_type == 'runup1', 'cohort_type'] = '2runup1'
Regression_Data.loc[Regression_Data.cohort_type == 'runup2', 'cohort_type'] = '3runup2'
Regression_Data.loc[Regression_Data.cohort_type == 'burst', 'cohort_type'] = '4burst'
Regression_Data.loc[Regression_Data.cohort_type == 'post', 'cohort_type'] = '5post'

Regression_Data = Regression_Data[Regression_Data.TRADE_DATE.between(20110000, 20200000)] # conditioining on specific time period

### Wealth Effect ###

Regression_Data['wealth_type'] = 'Institution'
Regression_Data.loc[Regression_Data.retail == 1, 'wealth_type'] = Regression_Data.loc[Regression_Data.retail == 1, 'group_label'].transform(lambda x: x.split("_")[1])

Regression_Data = Regression_Data[Regression_Data['wealth_type']!='0']
Regression_Data['Intercept'] = np.ones(Regression_Data.shape[0])
Regression_Data['TRADE_DATE'] = Regression_Data['TRADE_DATE'].astype('int')
Regression_Data = Regression_Data.set_index(['group_label','TRADE_DATE'])

# demand function estimation
res_list = []
dep_list = ['Profitability_win', 'Beta', 'dividend_be_win', 'log_BE', 'Age', 'INDUSTRYNAME_建筑装饰']
name_list = ['Profitability','Beta', 'Dividend', 'Size', 'Age', 'Construction Industry']
for y in dep_list:
    mod = PanelOLS.from_formula(y + ' ~ wealth_type + TimeEffects', Regression_Data)
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    res_list.append(res)

# save regression results
time_effects = ['Y'] *2
stock_effects = ['N'] * 2
result = compare_panel(res_list,stock_effects,time_effects, stars=True)
rows_main = deepcopy(result.summary.tables[0].data)
cols_to_keep = [1, 2] # columns to keep from the result
combined_rows = []

for r in rows_main:
    row_label = r[0]                    # use the left-most label
    others = [r[i] for i in cols_to_keep]
    combined_rows.append([row_label] + others)

combined_table = SimpleTable(combined_rows)
latex_output = combined_table.as_latex_tabular()

# format Latex output as lcc but not ccc
n_cols = len(combined_rows[0])
default_align = 'c' * n_cols
custom_align = 'l' + 'c' * (n_cols - 1)
latex_output = latex_output.replace(
    f"\\begin{{tabular}}{{{default_align}}}",
    f"\\begin{{tabular}}{{{custom_align}}}"
)
with open(os.path.join(TABLES_DIR, "demand_function_coefficients_wealth.tex"), "w") as f:
    f.write(latex_output)

# preferences/beliefs plotting

# marker to use
marker_list = 's'
width=0.25
color_list = ['tab:green']
base_x = np.arange(5)

for seq, res in enumerate(res_list):
    err_series = res.params - res.conf_int()['lower']
    coef_df = pd.DataFrame({'coef': res.params, 'err': err_series, 'varname': err_series.index.str.slice(12, -1)})

    fig, ax = plt.subplots(figsize=(12, 6))
    coef_df.plot(x='varname', y='coef', kind='bar', ax=ax, color='none', yerr='err', legend=False, ecolor=color_list[0])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=np.arange(coef_df.shape[0]), marker='s', s=120, y=coef_df['coef'], color=color_list[0])
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks(base_x)
    #     ax.xaxis.set_ticks_position('default')
    ax.set_xticklabels(['Wealth1(Low)','Wealth2', 'Wealth3', 'Wealth4','Wealth5(High)'], rotation=0, fontsize=18)
    ax.tick_params(axis='y', labelsize = 18)
    ax.set_title(f'Preferences/Beliefs about {name_list[seq]}', fontsize = 20)
    fig.savefig(os.path.join(FIGURES_DIR, f'demand_function_wealth_{name_list[seq]}.png'), dpi=300, bbox_inches='tight')

### Cohort ###

Regression_Data = beta_bubble[beta_bubble.retail == 1]
Regression_Data['cohort_type'] = 'Institution'
Regression_Data.loc[Regression_Data.retail == 1, 'cohort_type'] = Regression_Data.loc[Regression_Data.retail == 1, 'group_label'].transform(lambda x: x.split("_")[0])

Regression_Data.loc[Regression_Data.cohort_type == 'pre', 'cohort_type'] = '1pre'
Regression_Data.loc[Regression_Data.cohort_type == 'runup1', 'cohort_type'] = '2runup1'
Regression_Data.loc[Regression_Data.cohort_type == 'runup2', 'cohort_type'] = '3runup2'
Regression_Data.loc[Regression_Data.cohort_type == 'burst', 'cohort_type'] = '4burst'
Regression_Data.loc[Regression_Data.cohort_type == 'post', 'cohort_type'] = '5post'
Regression_Data.loc[Regression_Data.cohort_type == 'Institution', 'cohort_type'] = '6Institution'

Regression_Data = Regression_Data[Regression_Data['TRADE_DATE'].between(20140000,20170000)]

Regression_Data['Intercept'] = np.ones(Regression_Data.shape[0])
Regression_Data['TRADE_DATE'] = Regression_Data['TRADE_DATE'].astype('int')
Regression_Data = Regression_Data.set_index(['group_label','TRADE_DATE'])

res_list = []
var_list = ['Profitability_win', 'Beta', 'dividend_be_win','log_BE','Age','INDUSTRYNAME_建筑装饰']
name_list = ['Profitability', 'CAPM Beta', 'Dividend', 'Size', 'Age', 'Construction Industry']
for y in var_list:
    mod = PanelOLS.from_formula(y + ' ~ cohort_type+ TimeEffects', Regression_Data) # 
    res = mod.fit(cov_type='clustered', cluster_entity=True)
    res_list.append(res)

# save regression results
time_effects = ['Y'] *2
stock_effects = ['N'] * 2

result = compare_panel(res_list,stock_effects,time_effects, stars=True)
rows_main = deepcopy(result.summary.tables[0].data)

cols_to_keep = [1, 2] # columns to keep from the result

combined_rows = []

for r in rows_main:
    row_label = r[0]                    # use the left-most label
    others = [r[i] for i in cols_to_keep]
    combined_rows.append([row_label] + others)

combined_table = SimpleTable(combined_rows)
latex_output = combined_table.as_latex_tabular()

# format Latex output as lcc but not ccc
n_cols = len(combined_rows[0])
default_align = 'c' * n_cols
custom_align = 'l' + 'c' * (n_cols - 1)
latex_output = latex_output.replace(
    f"\\begin{{tabular}}{{{default_align}}}",
    f"\\begin{{tabular}}{{{custom_align}}}"
)

with open(os.path.join(TABLES_DIR, "demand_function_coefficients_cohort.tex"), "w") as f:
    f.write(latex_output)

# preferences/beliefs plotting

# marker to use
marker_list = 's'
width=0.25
color_list = ['tab:green']
# 6 types in total
base_x = np.arange(5)

for seq, res in enumerate(res_list):
    err_series = res.params - res.conf_int()['lower']
    coef_df = pd.DataFrame({'coef': res.params, 'err': err_series, 'varname': err_series.index.str.slice(12, -1)})

    fig, ax = plt.subplots(figsize=(12, 6))
    coef_df.plot(x='varname', y='coef', kind='bar', ax=ax, color='none', yerr='err', legend=False, ecolor=color_list[0])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(x=np.arange(coef_df.shape[0]), marker='s', s=120, y=coef_df['coef'], color=color_list[0])
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks(base_x)
    #     ax.xaxis.set_ticks_position('default')
    ax.set_xticklabels(['Pre-Bubble','Formation','Expansion','Deflation','Post-Bubble'], rotation=0, fontsize=15)
    ax.tick_params(axis='y', labelsize = 15)
    ax.set_title(f'Preferences/Beliefs about {name_list[seq]}', fontsize = 20)
    fig.savefig(os.path.join(FIGURES_DIR, f'demand_function_cohort_{name_list[seq]}.png'), dpi=300, bbox_inches='tight')