# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:54:29 2022

@author: rfremgen
"""


import os 
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import pyodbc
from sqlalchemy import create_engine
import mysql.connector as sql 

#%% Create new columns to check if a term is in a particular article

# Link key terms to where they appear in articles along with sentiment score 

term_save = ['nuclear', 'solar', 'wind', 'hydro', 'lng', 'natural gas',
             'coal', 'oil', 'carbon', 'emission', 'renewable', 'fossil fuel']

df_new = df_price.copy() 

for val in term_save:  
    
    df_new["term"] = df_new["snip"].map(lambda x: val if x is not None and val in x
                                        else (0 if x is not None else None))   
    
    df1 = df_new[df_new['term'] == val]  
    df1 = df1[col_save] 
    df_save.append(df1)

df_energy = pd.concat(df_save)

#%%

# Build data frame for Word Clous in Tableau 

import spacy 

def rmv_stop_words(data):
    new_data = data.copy() 
    en_model = spacy.load('en_core_web_sm')   
    stopwords = en_model.Defaults.stop_words 
    new_data['single'] = new_data['snip'].str.split()
    new_data['single'] = new_data['single'].apply(lambda x: [item for item in x 
                                                              if item not in stopwords 
                                                              and len(item) > 2
                                                              and len(item) < 17])
    new_data['single'] = [','.join(map(str, l)) for l in new_data['single']]
    new_data['single'] = new_data['single'].str.replace(',', ' ')
    return(new_data) 

def get_word_cloud_data(topic):
                      
    # Get data from database
    df_clean = df_company.dropna(axis=0, subset = ['snip'])
    
    df_positive = df_clean[df_clean['win_group'] == 'Positive'] 
    df_pos_stop = rmv_stop_words(df_positive)
    e = df_pos_stop.single.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    word_df2 = e.to_frame().reset_index()
    word_df2.rename({'index' : 'word', 0: 'count'}, axis=1, inplace=True) 
    word_df2['char_length'] = word_df2['word'].str.len() 
    word_df2 = word_df2.loc[(word_df2['char_length'] > 0)]  
    word_df2 = word_df2.sort_values('count', ascending=False).reset_index(drop=True).drop('char_length', axis=1) 
    word_df2 = word_df2[~word_df2['word'].isin(drop_words)] 
    word_df2 = word_df2[:100]
    word_df2['class'] = 'Positive'
    positive_df = word_df2
    positive_df['topic'] = 'Crown Castle' 
    
    
    df_negative = df_clean[df_clean['win_group'] == 'Negative'] 
    df_neg_stop = rmv_stop_words(df_negative)
    e = df_neg_stop.single.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    word_df2 = e.to_frame().reset_index()
    word_df2.rename({'index' : 'word', 0: 'count'}, axis=1, inplace=True) 
    word_df2['char_length'] = word_df2['word'].str.len() 
    word_df2 = word_df2.loc[(word_df2['char_length'] > 0)]  
    word_df2 = word_df2.sort_values('count', ascending=False).reset_index(drop=True).drop('char_length', axis=1) 
    word_df2 = word_df2[~word_df2['word'].isin(drop_words)] 
    word_df2 = word_df2[:100]
    word_df2['class'] = 'Negative'
    negative_df = word_df2
    negative_df['topic'] = 'Crown Castle'
    
    df_final = pd.concat([positive_df, negative_df], ignore_index=True)
    return(df_final)
