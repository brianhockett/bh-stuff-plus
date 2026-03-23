# Imports
import pandas as pd
import requests
import logging

# Initializing logger
logging.basicConfig(
    level = logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, and message format
    filename = 'generate-bio.log'                        # Log output file
)
logger = logging.getLogger(__name__)  # Create a logger instance

# Load Statcast data to get pitcher IDs
statcast_df = pd.read_parquet('./statcast-data.parquet')
print("Loaded in Statcast data")
logger.info("Loaded in Statcast data")

# Get unique pitcher IDs
pitcher_ids = statcast_df['pitcher'].dropna().unique()
print(f"Found {len(pitcher_ids)} unique pitchers")
logger.info(f"Found {len(pitcher_ids)} unique pitchers")

# List to hold biographical data
pitcher_bios = []

# Loop through each unique pitcher id
for i, pitcher_id in enumerate(pitcher_ids):
    if i % 200 == 0:
        print(f'Processing pitcher {i + 1} / {len(pitcher_ids)}')
        logger.info(f'Processing pitcher {i + 1} / {len(pitcher_ids)}')

    try:
        # Fetch biographical data from MLB Stats API
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={int(pitcher_id)}&hydrate=currentTeam"
        response = requests.get(url).json()

        # Extract player information
        player = response['people'][0]
    
        pitcher_bios.append({
            'pitcherID': int(pitcher_id),
            'name': player.get('fullName', ''),
            'throws': player.get('pitchHand', {}).get('code', ''),
            'age': player.get('currentAge', ''),
            'height': player.get('height', ''),
            'weight': player.get('weight', '')
        })
    except Exception as e:
        print(f"Error fetching data for pitcher {pitcher_id}: {e}")
        logger.error(f"Error fetching data for pitcher {pitcher_id}: {e}")

# Create DataFrame
df_bio = pd.DataFrame(pitcher_bios)

# Saving data to CSV and Parquet
# Save to CSV and Parquet
df_bio.to_csv('../data/pitcher-bio.csv', index = False)
df_bio.to_parquet('../data/pitcher-bio.parquet', index = False)

print("Saved biographical data as pitcher-bio.csv and pitcher-bio.parquet")
logger.info("Saved biographical data as pitcher-bio.csv and pitcher-bio.parquet")
