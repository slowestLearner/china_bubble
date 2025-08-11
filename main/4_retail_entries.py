#!/usr/bin/env python3.13.2
# -*- coding: utf-8 -*-
"""
What Drives Stock Prices in a Bubble?

Retial Entries

Created on Wed Jul 2 2025

Estimated running time: 5 sec
"""

import sys
import os

# Defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UTIL_DIR = os.path.join(os.path.join(BASE_DIR, 'code',"util"))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "output", "figures")
TABLES_DIR = os.path.join(BASE_DIR, "output", "tables")

sys.path.append(UTIL_DIR)
from general_import import *
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

from format_results import *

# === 1. Number of Investors on Entry ===

entry = pd.read_parquet(os.path.join(RESULTS_DIR, "flow_entry_data/new_entry"))
entry = entry.rename(columns={'Trade_Date':'TRADE_DATE'})
entry['TRADE_DATE'] = pd.to_datetime(entry['TRADE_DATE'], format='%Y%m%d')

### New Entries ###

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(entry['TRADE_DATE'],entry['new_acct']*5)

ax.set_xlabel('Time', fontsize = 15)
ax.set_ylabel('Number of New Investors', fontsize = 15)
# ax.set_title('Number of New Investors Entering the Market', fontsize = 17)
ax.tick_params(axis='y', labelsize = 15)
ax.tick_params(axis='x', labelsize = 15)
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
plt.axvline(x=pd.to_datetime(20140701, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20150101, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20150630, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20160131, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axhline(y=0, color='tab:orange', linestyle='--')

# plt.plot(pd.to_datetime(20150227, format='%Y%m%d'),43955*5,'ko')
# plt.annotate('X', (pd.to_datetime(20150227, format='%Y%m%d'),43955*5), xytext =(pd.to_datetime(20150327, format='%Y%m%d'),43955*4), fontsize=12)

fig.savefig(os.path.join(FIGURES_DIR, "retail_new_entries.png"), dpi=300, bbox_inches='tight')

plt.show()

# === 2. Calculate the time-series of capital net flows ===

# define big bubble stocks
big_bubble = set([600005, 600006, 600010, 600021, 600030, 600039, 600057, 600058, 600067, 600070, 600071, 600072, 600075, 600076, 600077, 600078, 600093, 600094, 600095, 600097, 600099, 600108, 600110, 600129, 600133, 600135, 600145, 600148, 600150, 600152, 600157, 600158, 600161, 600167, 600169, 600178, 600180, 600186, 600187, 600192, 600199, 600202, 600208, 600211, 600212, 600213, 600227, 600243, 600255, 600260, 600262, 600266, 600275, 600280, 600283, 600287, 600302, 600316, 600317, 600325, 600338, 600355, 600369, 600382, 600397, 600405, 600409, 600410, 600428, 600435, 600478, 600479, 600480, 600485, 600490, 600493, 600495, 600501, 600503, 600526, 600528, 600531, 600537, 600540, 600550, 600555, 600558, 600560, 600570, 600571, 600588, 600590, 600595, 600601, 600605, 600609, 600614, 600633, 600644, 600645, 600652, 600656, 600686, 600711, 600717, 600725, 600726, 600733, 600750, 600751, 600755, 600760, 600765, 600767, 600768, 600781, 600789, 600794, 600795, 600800, 600825, 600837, 600839, 600886, 600962, 600963, 600965, 600976, 600979, 600982, 600987, 600999, 601000, 601002, 601003, 601008, 601010, 601028, 601099, 601106, 601218, 601377, 601390, 601519, 601558, 601618, 601628, 601688, 601717, 601718, 601766, 601788, 601866, 601908, 601919, 601998])

# import the market cap. before bubble: denominator
retail_hold_stock = pd.read_parquet(os.path.join(DATA_DIR, "additional_results_from_return_chasing_paper/retail_share/retail_institution_hold_by_stock.parquet"))
retail_hold_stock['circulate_value'] = retail_hold_stock['Retail_Close_Value'] + retail_hold_stock['Institution_Close_Value']
market_cap = retail_hold_stock[['TRADE_DATE','SEC_CODE','circulate_value']]
market_cap_base = market_cap[market_cap.TRADE_DATE == 20140630]

# import Flow data
# import capital flow data from all retail investors (not sample)
netFlow_all = pd.read_parquet(os.path.join(RESULTS_DIR, "flow_entry_data/net_flow/type_net_flow_11_19"))
netFlow_all = netFlow_all.sort_values(['TRADE_DATE','SEC_CODE', 'ACCT_TYPE2','acct_attribute'])
netFlow_all = netFlow_all[~netFlow_all.ACCT_TYPE2.isnull()]
netFlow_all['retail'] = list(map(lambda x: int(x.startswith('1')), netFlow_all['ACCT_TYPE2'])) 
netFlow_all = netFlow_all[netFlow_all.retail == 1]

flow_cohort = pd.read_parquet(os.path.join(RESULTS_DIR, "flow_entry_data/cohort_netflow/"))

### Retail Net Flows ###

# target = sum(flow in big bubbles) / sum(big bubbles market cap)

net_flow = netFlow_all.copy()
net_flow['net_buy'] = net_flow['BUY_AMT'] - net_flow['SELL_AMT']
net_flow = net_flow[net_flow.SEC_CODE.apply(lambda x: x in big_bubble)]
net_flow['TRADE_DATE'] = net_flow['TRADE_DATE'].astype('int')
denominator = market_cap_base[market_cap_base.SEC_CODE.apply(lambda x: x in big_bubble)]['circulate_value'].sum()
net_flow_agg = net_flow.groupby(['TRADE_DATE', 'retail'])['net_buy'].sum().reset_index()
net_flow_agg['net_buy_share'] = net_flow_agg['net_buy'] / denominator * 100
    
fig, ax = plt.subplots(figsize=(12,6))
net_flow_agg = net_flow_agg.sort_values('TRADE_DATE')
net_flow_agg['TRADE_DATE'] = pd.to_datetime(net_flow_agg['TRADE_DATE'], format='%Y%m%d')
net_flow_agg = net_flow_agg.set_index('TRADE_DATE')
for col in net_flow_agg['retail'].unique():
    ax.plot(net_flow_agg[net_flow_agg['retail']==col]['net_buy_share'], label=f"{col}")
ax.set_xlabel('Time', fontsize = 15)
ax.set_ylabel(r'$\frac{Net Flow}{Market Cap. 06/2014}$ (%)', fontsize = 15)
#ax.set_title('Retail Investors\' Net Capital Flows Into Big Bubble Stocks', fontsize = 17)
ax.tick_params(axis='y', labelsize = 15)
ax.tick_params(axis='x', labelsize = 15)
#ax.legend()
plt.axvline(x=pd.to_datetime(20140701, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20150101, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20150630, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axvline(x=pd.to_datetime(20160131, format='%Y%m%d'), color='k', linestyle='--', alpha=0.6)
plt.axhline(y=0, color='tab:orange', linestyle='--')
    
#plt.plot(pd.to_datetime(20150227, format='%Y%m%d'),3.61,'ko')
#plt.annotate('X', (pd.to_datetime(20150227, format='%Y%m%d'),3.61), xytext =(pd.to_datetime(20150327, format='%Y%m%d'),3),fontsize=12)

fig.savefig(os.path.join(FIGURES_DIR, "retail_net_flows.png"), dpi=300, bbox_inches='tight')
plt.show()
