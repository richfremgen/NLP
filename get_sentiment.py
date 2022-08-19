# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:16:30 2022

@author: rfremgen
"""

# Load Packages
import os 
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import mysql.connector as sql 
from transformers import pipeline
import re
from transformers import BertForSequenceClassification, BertTokenizer
import torch

#%% Clean Data

def clean_data(data, phrase):
    
    """ Extracts snippets around phrase of interest """
    
    liner = [] 

    data['clean_url'] = data['clean_url'].str.split('.').str[0]   
    data['clean_url'] = data['clean_url'].str.capitalize() 
    data['title'] = data['title'].str.title() 
    data['art_len'] = data['summary'].str.len()  
    data = data.loc[data['art_len'] > 0].reset_index(drop=True) 
    sum_col =  list(data['summary']) 
    
    for val in sum_col:
        a = val.lower()
        a = a.replace(
            'inc.', '').replace(
                'story continues', '').replace(
                    'co.', '').replace(
                        'llc.', '').replace(
                            'corp.', '') 
                            
        b = re.split('[.]', a) 
        c = [b.index(s) for s in b if phrase in s] 
        one_line = [b[i] for i in c if len(b[i]) > 0] 
        liner.append('.'.join(one_line))
        
    #test = [i for i in liner if len(i) > 0] 
    data['snip'] = liner
    data = data.drop_duplicates(subset=['snip'])
    data = data.drop(['art_len', 'summary'], axis=1)
    data['words'] = data.snip.apply(lambda x: len(str(x).split(' '))) 
    data = data[data['words'] > 2]

    return(data.reset_index(drop=True)) 


#%%

def get_summary(string):
    
    """ Returns a condensed article summary given an input string """
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    ARTICLE = string
    summary =  summarizer(ARTICLE, max_length=200, min_length=5, 
                          do_sample=False, truncation=True)
    
    return(summary[0]['summary_text'])
    
#%% Get sentiment score 

def get_sentiment(string): 
    
    """ Returns sentiment prediction and probability for news snippet """
    
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')
    
    tokens = tokenizer.encode_plus(string, add_special_tokens=False, return_tensors='pt') 
    # input_id_chunks = tokens['input_ids'][0].split(510)
    # mask_chunks = tokens['attention_mask'][0].split(510)
    
    # define target chunksize
    chunksize = 512
    
    # split into chunks of 510 tokens, we also convert to list (default is tuple which is immutable)
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))
    
    # loop through each chunk
    for i in range(len(input_id_chunks)):
        # add CLS and SEP tokens to input IDs 
        input_id_chunks[i] = torch.cat([
            torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
        ])
        # add attention tokens to attention mask
        mask_chunks[i] = torch.cat([
            torch.tensor([1]), mask_chunks[i], torch.tensor([1])
        ])
        # get required padding length
        pad_len = chunksize - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])
    
    # check length of each tensor
    # for chunk in input_id_chunks:
    #     print(len(chunk))
    
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)
    
    input_dict = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.int()
    } 
    
    outputs = model(**input_dict)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1) 
    # print(probs)
    probs = probs.mean(dim=0)
   # print(probs)
    winner = torch.argmax(probs).item() 
    output_dict = {"prob": probs, "winner": winner}
   # print(['positive', 'negative', 'neutral'][winner])  
    return(output_dict)

#%%

def get_results(snip_list, start, end, data): 
    
    save_summary = []
    save_pos = [] 
    save_neg = []
    save_neutral = []
    save_winner = []
    
    trim_list = snip_list[start:end]
    trim_df = data[start:end].reset_index(drop=True)
    
    for i,art in enumerate(trim_list): 
        
        p = get_summary(art)
        k = get_sentiment(p)
        save_summary.append(p) 
        save_winner.append(k['winner']) 
        save_pos.append(k['prob'].tolist()[0]) 
        save_neg.append(k['prob'].tolist()[1]) 
        save_neutral.append(k['prob'].tolist()[2]) 
        print(i)
        
    results_df = pd.DataFrame({
        'summary': save_summary,
        'winner' : save_winner,
        'positive' : save_pos,
        'negative' : save_neg,
        'neutral' : save_neutral}).reset_index(drop=True)
    
    combined_df = pd.concat([trim_df, results_df], axis=1)
    return(combined_df) 


