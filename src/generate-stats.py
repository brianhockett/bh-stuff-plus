# Imports
from pybaseball import playerid_reverse_lookup
import pandas as pd
import requests
import logging

# Initializing logger
logging.basicConfig(
    level = logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, and message format
    filename = 'generate-stats.log'                      # Log output file
)
logger = logging.getLogger(__name__)  # Create a logger instance

# Fetch data from Fangraphs API for the 2021-2025 MLB seasons
years = [2021, 2022, 2023, 2024, 2025]
all_data = []

# Loop through each year
for year in years:
    print(f"Fetching Fangraphs statistics data for {year}")
    logger.info(f"Fetching Fangraphs statistics data for {year}")

    # Sending API call for iteration year
    response = requests.get(
        f"https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season={year}&ind=1&qual=0&type=8&month=0&pageitems=500000"
    ).json()
    
    # If the data come back correctly, append it to the total data
    if 'data' in response:
        all_data.extend(response['data'])

# Create pandas dataframe
df = pd.DataFrame(all_data)

# Only select columns of interest
df = df[['xMLBAMID', 'playerid', 'Season', 'PlayerName', 'Age', 
         'Throws', 'ERA', 'FIP', 'xFIP', 
         'IP', 'Pitches', 'WAR']]

# Rename mlbam id
df = df.rename(columns = {'xMLBAMID': 'pitcherID'})

# Reorder columns
df = df[['pitcherID', 'playerid', 'Season', 'PlayerName', 'Age', 'Throws', 'ERA', 'FIP', 'xFIP', 'IP', 'Pitches', 'WAR']]

# Add pitcherID_Season primary key column
df.insert(0, 'pitcherID_Season', df['pitcherID'].astype(str) + '_' + df['Season'].astype(str))

# Saving to CSV and Parquet
df.to_csv('../data/pitcher-stats.csv', index = False)
df.to_parquet('../data/pitcher-stats.parquet', index = False)

print("Saved pitcher statistical data as pitcher-stats.csv and pitcher-stats.parquet")
logger.info("Saved pitcher statistical data as pitcher-stats.csv and pitcher-stats.parquet")