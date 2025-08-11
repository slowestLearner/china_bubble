#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
from tqdm import tqdm
import os
import pandas as pd
from pathlib import Path

# # combine_csv_files

# In[2]:


def combine_csv_files(file_path, save_path, skip_header = True):
    fout = open(save_path,"a")
    all_files = glob.glob(file_path)
    for line in open(all_files[0]):
        fout.write(line)
    for file in tqdm(all_files[1:]):
        f = open(file)
        f.__next__() # skip the header
        for line in f:
            fout.write(line)
        f.close() # not really needed
    fout.close()
    return print("Finished")


# # shut_down

# In[2]:


def shut_down():
    os.system('shutdown -s -t 3')


# #  convert data type

# In[1]:


def convert_dtype(df, cols, data_type):
    for col in cols:
        df[col] = df[col].astype(data_type)
    return df


def convert_date(df, date_col, num_col):
    df[num_col] = df[date_col].str.replace('-','').apply(int)
    return df
    

def convert_num_to_date(df, num_col, date_col):
    df[date_col] = pd.to_datetime(df[num_col].astype("str"), format='%Y-%m-%d', errors='coerce')
    return df


def if_month_end(file_date, next_file_date):
    '''
    This function checks if two dates are in different months.
    '''
    file_date = str(file_date)
    next_file_date = str(next_file_date)
    now_month = int(file_date[4:6])
    next_month = int(next_file_date[4:6])
    now_year = int(file_date[0:4])
    next_year = int(next_file_date[0:4])
    if (now_month == next_month) & (now_year == next_year):
        return False
    else:
        return True
    
def generate_col_names(col_names):
    col_names = col_names.split(',')
    col_names = [x.strip() for x in col_names]
    return col_names

def get_file_name(file_date, data_set):
    '''
    Find the file name based on the trade date.
    '''
    file_date = str(file_date)
    for single_file in data_set:
        trade_date = single_file.split("\\")[-1].split(".")[0].split("_")[-1]
        if file_date == trade_date:
            return single_file
    return print(file_date + "_cannot_found_the_daily_file")

def get_file_date(file_name):
    '''
    Returns the last date suffix of the file name.
    '''
    file_date = file_name.split("\\")[-1].split(".")[0].split("_")[-1]
    return file_date


def combine_filelist(file_list, trade_date=False, parquet= False):
    daily_data = []
    for _ in file_list:
        if parquet:
            temp = pd.read_parquet(_)
        else:
            temp = pd.read_csv(_)
        if trade_date:
            file_date = _.split("\\")[-1].split("_")[-1]
            temp['TRADE_DATE'] = int(file_date)
        daily_data.append(temp)
    daily_data = pd.concat(daily_data, ignore_index=True)
    return daily_data


def get_month_file(month_begin, month_end,file_list):
    '''
    Obtains the file names for the current month based on the trade dates.
    '''
    month_file_list = []
    for daily_file in file_list:
        trade_date = int(daily_file.split("\\")[-1].split("_")[-1])
        if (trade_date > month_begin) & (trade_date<= month_end):
            month_file_list.append(daily_file)
    return month_file_list
    

def create_path(save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)





