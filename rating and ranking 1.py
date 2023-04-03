# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 19:40:25 2023

@author: HAMILJ37
"""
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import statistics as stat
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
import numpy.linalg as linalg
from sklearn.preprocessing import normalize
from numpy.linalg import eig

wg = pd.read_csv('/2022NFLseason.csv')
wg = wg.rename(columns={'Unnamed: 5': 'homeaway','Winner/tie':'winner','Loser/tie':'loser'})
wg = wg.drop('Unnamed: 7', axis = 1)

for lab, row in wg.iterrows():
    if wg.loc[lab,'homeaway'] == '@':
        wg.loc[lab,'homeaway'] = -2
    else:
        wg.loc[lab,'homeaway'] = 2
wg = wg.dropna()
wg = wg.reset_index(drop = True)
        
wg['diff'] = wg.PtsW - wg.PtsL


wg_reg_season = wg[0:271]
wg_reg_season['Week'] = wg_reg_season['Week'].astype(np.int64)

p_matrix = wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'diff').fillna(0)  - wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'diff').fillna(0).T

P = wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'diff', aggfunc = 'count').fillna(0) + wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'diff', aggfunc = 'count').fillna(0).T 
num_teams = len(pd.unique(wg_reg_season[['winner','loser']].to_numpy().flatten()))
T = pd.DataFrame(np.identity(n = num_teams), index = P.index, columns = P.columns)
for i, row in T.iterrows():
    num_games_played = len(wg_reg_season[wg_reg_season.winner == i]) + len(wg_reg_season[wg_reg_season.loser == i])
    T.loc[i,i] = num_games_played
    
M = T - P
    
# let's solve for the basic massey method - Mr = p

# worrying about full rank isn't necessary for this year because we're missing a game
# M_full_rank = M.copy()
# M_full_rank = M_full_rank.T
# M_full_rank[M_full_rank.index[0]] = 1
# M_full_rank = M_full_rank.T
# M_minus_one = linalg.inv(M_full_rank.values)
# p = wg_pivot_score.sum(axis = 1).values
# p[0] = 0
# r = linalg.lstsq(M,p)

p = p_matrix.sum(axis = 1).values
r = linalg.lstsq(M,p)[0]

ratings_df = M.index.to_frame()

ratings_df['wins'] = 0
ratings_df['losses'] = 0
ratings_df['ties'] = 0

for lab, row in ratings_df.iterrows():
    winners_df = wg_reg_season[wg_reg_season.winner == lab]
    losers_df = wg_reg_season[wg_reg_season.loser == lab]
    tiesw = len(winners_df[winners_df.PtsW == winners_df.PtsL])
    tiesl = len(losers_df[losers_df.PtsW == losers_df.PtsL])
    ratings_df.loc[lab,'wins'] = len(winners_df) - tiesw
    ratings_df.loc[lab,'losses'] = len(losers_df) - tiesl
    ratings_df.loc[lab,'ties'] = tiesw + tiesl

ratings_df['massey'] = r.tolist()

# massey offense and defense
f_matrix =  wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'PtsW').fillna(0)  + wg_reg_season.pivot_table(index = 'winner', columns = 'loser', values = 'PtsL').fillna(0).T
a_matrix = f_matrix - p_matrix
f = f_matrix.sum(axis = 1).values
a = a_matrix.sum(axis = 1).values

d = linalg.lstsq(T + P,T@r - f)[0]

o = r - d

ratings_df['massey_o'] = o.tolist()
ratings_df['massey_d'] = d.tolist()

# colley method

T_colley = pd.DataFrame(np.identity(n = num_teams), index = P.index, columns = P.columns)
for i, row in T_colley.iterrows():
    num_games_played = len(wg_reg_season[wg_reg_season.winner == i]) + len(wg_reg_season[wg_reg_season.loser == i])
    T_colley.loc[i,i] = num_games_played + 2
    
C = T_colley - P

b = ((ratings_df['wins'] - ratings_df['losses'])*.5 + 1).values

r_colley = linalg.lstsq(C, b)[0]

ratings_df['colley'] = r_colley.tolist()

# colleyized massey method C = M + 2I

r_mc_combo = linalg.lstsq(2*pd.DataFrame(np.identity(n = num_teams), index = P.index, columns = P.columns) + M, p)[0]

ratings_df['massey_colley_combo'] = r_mc_combo.tolist()

# elo

ratings_elo_time = pd.DataFrame(0, index = ratings_df.index, columns = range(0,19)).astype(float)
epsilon = 1000
k = 32
for week in range(1,19):
    ratings_elo_time_series = ratings_elo_time[week-1].copy()
    games_week = wg_reg_season[wg_reg_season.Week == week]
    for lab, row in games_week.iterrows():
        i = row['winner']
        j = row['loser']
        dij = ratings_elo_time_series[i] - ratings_elo_time_series[j]
        dji = -1 * dij
        muij = 1/(1+(pow(10,-dij/epsilon)))
        muji = 1/(1+(pow(10,-dji/epsilon)))
        ri_old = ratings_elo_time_series[i]
        rj_old = ratings_elo_time_series[j]
        if row['diff'] == 0:
            Sij = .5
            Sji = .5
        else:
            Sij = 1
            Sji = 0
        ratings_elo_time_series[i] = ri_old + k*(Sij - muij)
        ratings_elo_time_series[j] = rj_old + k*(Sji - muji)
    ratings_elo_time[week] = ratings_elo_time_series
ratings_df['elo'] = ratings_elo_time_series

ratings_elo_time.to_csv('elo1.csv')

# notes: comparing to fivethirtyeight
# fivethirtyeight uses priors
# normalized to 1500
# top QB thing

# elo 2- proportion of points scored- 

ratings_elo_time_points = pd.DataFrame(0, index = ratings_df.index, columns = range(0,19)).astype(float)
epsilon = 1000
for week in range(1,19):
    if week in [1,17,18]: k = 16
    else: k = 32
    ratings_elo_time_series = ratings_elo_time_points[week-1].copy()
    games_week = wg_reg_season[wg_reg_season.Week == week]
    for lab, row in games_week.iterrows():
        i = row['winner']
        j = row['loser']
        dij = ratings_elo_time_series[i] - ratings_elo_time_series[j]
        dji = -1 * dij
        muij = 1/(1+(10**(-dij/epsilon)))
        muji = 1/(1+(10**(-dji/epsilon)))
        ri_old = ratings_elo_time_series[i]
        rj_old = ratings_elo_time_series[j]
        Sij = row['PtsW'] / (row['PtsW'] + row['PtsL'])
        Sji = 1 - Sij
        ratings_elo_time_series[i] = ri_old + k*(Sij - muij)
        ratings_elo_time_series[j] = rj_old + k*(Sji - muji)
    ratings_elo_time_points[week] = ratings_elo_time_series
ratings_df['elo_points'] = ratings_elo_time_series