import pandas as pd 

df_stats = pd.read_csv('data/stats.csv', index_col='stat')

Z_SCORE = df_stats.loc['z_score'].values[0]
STD = df_stats.loc['std'].values[0]

def getInterval(pred):
    return (pred - Z_SCORE*STD, pred + Z_SCORE*STD)

