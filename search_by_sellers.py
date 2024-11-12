#%%
import ruten_api
import pandas as pd
import os

output_dir = 'output/search_results_by_sellers/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

top_sellers_dir = 'output/top_sellers/'
if not os.path.exists(top_sellers_dir):
    raise Exception('Top sellers data not found. Please run find_top_sellers.py first.')

#%%
# Load top sellers data and queries pool

top_sellers = {}
for file in os.listdir(top_sellers_dir):
    query = file.replace('_top_sellers.csv', '')
    top_sellers[query] = pd.read_csv(f'{top_sellers_dir}{file}')

queries_pool = list(top_sellers.keys())

# %%
# Search items by sellers for each query

for query in queries_pool:
    sellers = top_sellers[query]
    for i, row in sellers.iterrows():
        seller_nick = row['user']
        results = ruten_api.search(query, top_k=40, seller_nick=seller_nick, verbose=True)
        results.to_csv(f'{output_dir}{query}_{seller_nick}.csv', index=False)

# %%