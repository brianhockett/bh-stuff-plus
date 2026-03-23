# Imports
import pybaseball as pb
import pandas as pd
import warnings
import logging

# Ignore warnings (Removes warning about datetime format, script works as is)
warnings.filterwarnings('ignore')

# Initializing logger
logging.basicConfig(
    level = logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, level, and message format
    filename = 'generate-statcast.log'                   # Log output file
)
logger = logging.getLogger(__name__)  # Create a logger instance

# Years of data to generate
years = [2021, 2022, 2023, 2024, 2025]

# Month ranges for each year (Regular Season)
months_by_year = {
    2021: [('2021-04-01', '2021-04-30'), ('2021-05-01', '2021-05-31'),
           ('2021-06-01', '2021-06-30'), ('2021-07-01', '2021-07-31'),
           ('2021-08-01', '2021-08-31'), ('2021-09-01', '2021-09-30'),
           ('2021-10-01', '2021-10-03')],
    
    2022: [('2022-04-07', '2022-04-30'), ('2022-05-01', '2022-05-31'),
           ('2022-06-01', '2022-06-30'), ('2022-07-01', '2022-07-31'),
           ('2022-08-01', '2022-08-31'), ('2022-09-01', '2022-09-30'),
           ('2022-10-01', '2022-10-05')],
    
    2023: [('2023-03-30', '2023-04-30'), ('2023-05-01', '2023-05-31'),
           ('2023-06-01', '2023-06-30'), ('2023-07-01', '2023-07-31'),
           ('2023-08-01', '2023-08-31'), ('2023-09-01', '2023-09-30'),
           ('2023-10-01', '2023-10-01')],
    
    2024: [('2024-03-20', '2024-04-30'), ('2024-05-01', '2024-05-31'),
           ('2024-06-01', '2024-06-30'), ('2024-07-01', '2024-07-31'),
           ('2024-08-01', '2024-08-31'), ('2024-09-01', '2024-09-30')],
    
    2025: [('2025-03-18', '2025-03-31'), ('2025-04-01', '2025-04-30'),
           ('2025-05-01', '2025-05-31'), ('2025-06-01', '2025-06-30'),
           ('2025-07-01', '2025-07-31'), ('2025-08-01', '2025-08-31'),
           ('2025-09-01', '2025-09-28')]
}

# List to hold cumulative data
all_years_data = []

# Removing columns unrelated to analysis
columns_to_keep = ['pitch_type', 'game_date', 'game_type', 'release_speed',
                    'release_pos_x', 'release_pos_z',
                    'player_name', 'pitcher',
                    'events', 'description', 'p_throws',
                    'balls', 'strikes',
                    'game_year', 'pfx_x', 'pfx_z',
                    'ax', 'ay', 'az', 'release_spin_rate',
                    'release_extension', 'release_pos_y', 'pitch_name',
                    'spin_axis', 'delta_run_exp']

# Loop through each season
for year in years:
    print(f'Downloading Statcast data for {year}')
    logger.info(f'Downloading Statcast data for {year}')

    # List to hold current year's data
    year_data = []

    # Looping through each month in this iterations year
    for start, end in months_by_year[year]:
        try:
            print(f"Downloading data from {start} to {end}")
            logger.info(f"Downloading data from {start} to {end}")
            # Get monthly chunk of statcast data
            chunk = pb.statcast(start, end, verbose = False)
            if chunk is not None and len(chunk) > 0:
                chunk = chunk[columns_to_keep]
                year_data.append(chunk)
            else:
                print(f"No data found for {start} to {end}")
                logger.info(f"No data found for {start} to {end}")
        except Exception as e:
            print(f"Problem downloading data from {start} to {end}")
            logger.error(f"Problem downloading data from {start} to {end}")
    
    # Concatenate this years data into cumulative total
    if year_data:
        statcast_data_year = pd.concat(year_data, ignore_index = True)
        all_years_data.append(statcast_data_year)
        print(f"Completed downloading and appending {year}")
        logger.info(f"Completed downloading and appending {year}")
    else:
        print(f"No data for {year}")
        logger.info(f"No data for {year}")

# Concatenate all years into one file
if all_years_data:
    print("Combining all data from 2021-2025")
    logger.info("Combining all data from 2021-2025")

    # Concatenating all data into Pandas DataFrame
    statcast_data = pd.concat(all_years_data, ignore_index = True)

    # Keep only regular season games
    statcast_data = statcast_data[statcast_data['game_type'] == 'R']

    # Drop game_type now
    statcast_data = statcast_data.drop(columns = ['game_type'])

    # Drop automatic balls/strikes, no pitch characteristics
    statcast_data = statcast_data[~statcast_data['description'].isin(['automatic_ball', 'automatic_strike'])]

    # Convert break values to inches
    statcast_data['pfx_z'] = statcast_data['pfx_z'] * 12
    statcast_data['pfx_x'] = statcast_data['pfx_x'] * 12

    # Dictionary to simplify descriptions
    des_dict = {
    'hit_into_play': 'hit_into_play',
    'ball': 'ball',
    'called_strike': 'called_strike',
    'swinging_strike': 'swinging_strike',
    'swinging_strike_blocked': 'swinging_strike',
    'foul': 'foul',
    'foul_tip': 'swinging_strike',
    'foul_bunt': 'foul',
    'bunt_foul_tip': 'swinging_strike',
    'missed_bunt': 'swinging_strike',
    'blocked_ball': 'ball',
    'hit_by_pitch': 'hit_by_pitch',
    'pitchout': 'ball',
    'foul_pitchout': 'foul'
    }

    # Dictionary to simplify events
    event_dict = {  
    'single': 'single',
    'double': 'double',
    'triple': 'triple',
    'home_run': 'home_run',
    'walk': 'walk',
    'intent_walk': 'walk',
    'hit_by_pitch': 'hit_by_pitch',
    'strikeout': 'strikeout',
    'strikeout_double_play': 'strikeout',
    'field_out': 'field_out',
    'force_out': 'field_out',
    'grounded_into_double_play': 'field_out',
    'double_play': 'field_out',
    'triple_play': 'field_out',
    'fielders_choice': 'field_out',
    'fielders_choice_out': 'field_out',
    'sac_fly': 'field_out',
    'sac_fly_double_play': 'field_out',
    'sac_bunt': 'field_out',
    'sac_bunt_double_play': 'field_out',
    'other_out': 'field_out',
    'field_error': None,
    'catcher_interf': None,
    'truncated_pa': None,
    }

    # Mapping values
    statcast_data['description_group'] = statcast_data['description'].map(des_dict)
    statcast_data['events_group'] = statcast_data['events'].map(event_dict)

    # Drop anomalous data error found earlier, where desc was hit by pitch, but no event was recorded for unknown reason
    statcast_data = statcast_data[
    ~((statcast_data['description'] == 'hit_by_pitch') & (statcast_data['events'].isna()))
]

    # Drop anomalous walks on impossible ball counts that were found
    statcast_data = statcast_data[
        ~((statcast_data['events'] == 'walk') & (statcast_data['balls'] < 3))
    ]
    
    # Null out description_group for events with no clean run value mapping
    statcast_data.loc[statcast_data['events'].isin(['field_error', 'catcher_interf', 'truncated_pa']), 'description_group'] = None

    # Reordering dataframe for neatness
    statcast_data = statcast_data[['pitcher', 'player_name', 'p_throws', 
                                   'game_date', 'game_year', 'balls', 
                                   'strikes', 'pitch_type', 'pitch_name', 
                                   'release_speed', 'release_pos_x', 'release_pos_y', 
                                   'release_pos_z', 'release_extension', 'release_spin_rate', 
                                   'spin_axis', 'pfx_x', 'pfx_z', 
                                   'ax', 'ay', 'az', 
                                   'description', 'description_group', 
                                   'events', 'events_group',
                                   'delta_run_exp']]

    # Creating foreign-key map to Run Expectancy Table
    statcast_data['event_balls_strikes'] = (
        statcast_data['events_group'].combine_first(statcast_data['description_group']) + '_' +
        statcast_data['balls'].astype(str) + '_' + 
        statcast_data['strikes'].astype(str)
    )

    # Create primary key
    statcast_data.insert(0, 'pitchID', range(len(statcast_data)))

    # Add foreign key to join with stats table
    statcast_data.insert(6, 'pitcher_season', statcast_data['pitcher'].astype(int).astype(str) + '_' + statcast_data['game_year'].astype(str))

    # Saving data to CSV and Parquet
    statcast_data.to_parquet('../data/statcast-data.parquet', index = False)
    statcast_data.to_csv('../data/statcast-data.csv', index = False)

    print("Saved Statcast data as statcast-data.csv and statcast-data.parquet")
    logger.info("Saved Statcast data as statcast-data.csv and statcast-data.parquet")