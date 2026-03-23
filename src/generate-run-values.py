# Imports
import pandas as pd
import logging

# Initializing logger
logging.basicConfig(
    level = logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, and message format
    filename = 'generate-run-values.log'                 # Log output file
)
logger = logging.getLogger(__name__)  # Create a logger instance

# Load Statcast data
statcast_df = pd.read_parquet('./statcast-data.parquet')
print("Loaded in Statcast data")
logger.info("Loaded in Statcast data")

# Generate run values for PA-ending pitches (event-based)
print("Generating event-based run values")
logger.info("Generating event-based run values")
rv_events = (
    statcast_df[statcast_df['events_group'].notna()]
    .groupby(['events_group', 'balls', 'strikes'])['delta_run_exp']
    .mean()
    .reset_index()
    .rename(columns = {'events_group': 'event'})
)

# Generate run values for non-PA-ending pitches (description-based)
print("Generating description-based run values")
logger.info("Generating description-based run values")
rv_desc = (
    statcast_df[statcast_df['events_group'].isna() & statcast_df['description_group'].notna()]
    .groupby(['description_group', 'balls', 'strikes'])['delta_run_exp']
    .mean()
    .reset_index()
    .rename(columns = {'description_group': 'event'})
)

# Combine into a single run values table
print('Combining run values into single table')
logger.info('Combining run values into single table')
run_values = pd.concat([rv_events, rv_desc], ignore_index = True)

# Create primary key
run_values.insert(0, 'event_balls_strikes', 
                  run_values['event'] + '_' + 
                  run_values['balls'].astype(str) + '_' + 
                  run_values['strikes'].astype(str))

# Drop null delta_run_exp (intentional walk intermediate pitches)
run_values = run_values.dropna(subset=['delta_run_exp'])

# Saving to CSV and Parquet
run_values.to_csv('../data/run-values.csv', index = False)
run_values.to_parquet('../data/run-values.parquet', index = False)

print("Saved run values as run-values.csv and run-values.parquet")
logger.info("Saved run values as run-values.csv and run-values.parquet")