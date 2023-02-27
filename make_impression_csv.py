import pandas as pd

df = pd.read_csv('~/rads_dispo_lim_2023_02_23.csv')
df = df[['Impression']].rename(columns={"Impression": "Report Impression"})
df.to_csv('~/impressions.csv', index=False)
