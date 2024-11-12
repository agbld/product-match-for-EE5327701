#%%
import ruten_api
import pandas as pd
import os

output_dir = 'output/top_sellers/' # Output directory

num_sellers = 2 # Number of sellers to find for each query
top_k = 200 # Number of items to use for justifying top sellers

# Load queries from queries.txt
with open('queries.txt', 'r', encoding='utf-8') as file:
    queries_pool = [line.strip() for line in file.readlines()]

#%%
# Find sellers with most items for each query

top_sellers = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for query in queries_pool:
    results = ruten_api.search(query, top_k, verbose=True)
    sellers = results.groupby('user').size().sort_values(ascending=False).reset_index(name='count')
    sellers = sellers.head(num_sellers)
    top_sellers[query] = sellers

# Save top_sellers to csv
for query, sellers in top_sellers.items():
    sellers.to_csv(f'{output_dir}{query}_top_sellers.csv', index=False)

#%%