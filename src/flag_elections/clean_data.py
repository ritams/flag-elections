import pandas as pd
import numpy as np
import pickle
import os

# Define paths
RAW_DATA_PATH = os.path.join("data", "raw")
CLEANED_DATA_PATH = os.path.join("data", "cleaned")

# Verify input files exist
file_a_j = os.path.join(RAW_DATA_PATH, 'clea_lc_20220908 A-J.xlsx')
file_k_z = os.path.join(RAW_DATA_PATH, 'clea_lc_20220908 K-Z.xlsx')

if not os.path.exists(file_a_j) or not os.path.exists(file_k_z):
    raise FileNotFoundError(f"Missing raw data files in {RAW_DATA_PATH}")

print("Loading Excel files...")
df1 = pd.read_excel(file_a_j)
df2 = pd.read_excel(file_k_z)

print("Concatenating dataframes...")
df = pd.concat([df1, df2], ignore_index=True)

# Create output directory if it doesn't exist
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

all_countries = df['ctr_n'].unique()
print(f"Found {len(all_countries)} countries.")

vote_dict_all = {}

print("Processing countries...")
for country in all_countries:
    print(f"Processing {country}...")
    df_country = df[df['ctr_n'] == country]
    years = df_country['yr'].unique()
    
    vote_dict = {}
    for year in years:
        vote_dict[year] = []
        df_year = df_country[df_country['yr'] == year]
        constituencies = df_year['cst'].unique()
        
        for c in constituencies:
            df_c = df_year[df_year['cst'] == c]
            # Handle potential NaN values in cv1
            cv1_c = df_c['cv1'].dropna().values
            
            if len(cv1_c) == 0:
                continue
                
            votes = np.zeros(len(cv1_c), dtype=int)
            for i in range(len(cv1_c)):
                 if cv1_c[i] > 0:
                     votes[i] += int(cv1_c[i])
            
            # Sort votes in descending order for consistency with the snippet logic (though logic uses sort later)
            # The snippet `vote_dict[year].append(sorted(votes))` sorts ascending.
            # Wait, snippet says: `vote_dict[year].append(sorted(votes))`
            vote_dict[year].append(sorted(votes))
            
    vote_dict_all[country] = vote_dict

# Save vote_dict_all
vote_dict_path = os.path.join(CLEANED_DATA_PATH, 'vote_dict_all.pkl')
print(f"Saving {vote_dict_path}...")
with open(vote_dict_path, 'wb') as f:
    pickle.dump(vote_dict_all, f)

# Create readme for vote_dict_all
readme_path = os.path.join(CLEANED_DATA_PATH, 'readme.md')
with open(readme_path, 'w') as f:
    text = """
filename: vote_dict_all.pkl
The data is in the format of data[country][year][constituency_index] = [v1, v2, v3, ...]
"""
    f.write(text)

# Calculate effective number of candidates and other metrics
print("Calculating election metrics...")
cleaned_election_data = {}

for country in vote_dict_all:
    cleaned_election_data[country] = {
        'turnout': [],
        'winner': [],
        'runner_up': [],
        'eff_num': [],
        'vote_list': [],
        'eff_num_float': []
    }
    
    for year in vote_dict_all[country]:
        year_data = vote_dict_all[country][year]
        
        for constituency_votes in year_data:
            # votes are already lists from the previous step
            vote_list = np.sort(np.nan_to_num(constituency_votes))[::-1] # descend
            
            turnout = np.sum(vote_list)
            
            if turnout > 0 and len(vote_list) > 2:
                eff_num_float = 1.0 / np.sum((vote_list / turnout) ** 2)
                eff_num = max(2, round(eff_num_float))
                
                cleaned_election_data[country]['turnout'].append(turnout)
                cleaned_election_data[country]['winner'].append(vote_list[0])
                cleaned_election_data[country]['runner_up'].append(vote_list[1])
                cleaned_election_data[country]['eff_num'].append(eff_num)
                cleaned_election_data[country]['vote_list'].append(vote_list)
                cleaned_election_data[country]['eff_num_float'].append(eff_num_float)

# Save cleaned_election_data
cleaned_data_path = os.path.join(CLEANED_DATA_PATH, 'cleaned_election_data.pkl')
print(f"Saving {cleaned_data_path}...")
with open(cleaned_data_path, 'wb') as f:
    pickle.dump(cleaned_election_data, f)

# Append to readme
with open(readme_path, 'a') as f:
    text = """
filename: cleaned_election_data.pkl
The data is in the format of data[country] = {'turnout': [], 'winner': [], 'runner_up': [], 'eff_num': [], 'vote_list': [], 'eff_num_float': []}
"""
    f.write(text)

print("Done.")
