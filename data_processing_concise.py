#data_processing_concise.py


import numpy as np            # For numerical operations (e.g., np.abs and np.nan)
import pandas as pd           # For handling DataFrames
from datetime import timedelta # For handling time intervals

# Function to calculate absolute and fractional deviations



def generate_parameter_list(varying_param, varying_values, constant_params):
        """
        Generates a list of parameter dictionaries where one parameter varies.

        Parameters:
            varying_param (str): The parameter that will vary.
            varying_values (list): A list of values for the varying parameter.
            constant_params (dict): A dictionary of constant parameter values.

        Returns:
            list: A list of dictionaries with the specified varying parameter and constant parameters.
        """
        parameter_list = [
            {**constant_params, varying_param: value} for value in varying_values
        ]
        return parameter_list


def calculate_deviations(df):
    df['ab_deviation'] = np.abs(df['pm25_A'] - df['pm25_B'])
    df['fraction_deviation'] = df['ab_deviation'] / df['pm25_A']
    df['check'] = (df['ab_deviation'] > 5.0) & (df['fraction_deviation'] > 0.7)
    df.loc[df['check'], 'pm25_avg'] = np.nan
    df.loc[~df['check'], 'pm25_avg'] = df[['pm25_A', 'pm25_B']].mean(axis=1)

    df['new_corrected'] = df['pm25_avg'].apply(lambda x: max(0, 0.79 * x - 7.96) if pd.notna(x) else np.nan)



def get_concentration_stats(start_time, end_time, df):
    # Work with a copy of df to avoid modifying the original
    df_copy = df.copy()
    
    df_copy['datetime_utc'] = pd.to_datetime(df_copy['datetime_utc'])
    df_copy = df_copy.set_index('datetime_utc')

    mask = (df_copy.index >= start_time) & (df_copy.index <= end_time)
    filtered_data = df_copy.loc[mask, 'new_corrected']
    average_concentration = filtered_data.mean()
    median_concentration = filtered_data.median()
    maximum_concentration = filtered_data.max()

    if average_concentration == 0:
        print("Average concentration is 0 for the following section of the DataFrame:")
        print(df_copy.loc[mask]) 
    
    return average_concentration, median_concentration, maximum_concentration



def rolling_window_event_data(dfs, Window):
    
    all_dfs_events = []
    num_30min_intervals = int(Window*2)
    

    for df in dfs:
        # Ensure 'datetime_utc' is a datetime type for proper processing
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df.set_index('datetime_utc')

        # Prepare a dictionary to collect all metrics
        metrics = {
            

            'num_intervals_over_level_1_threshold_concentration': [],
            'num_intervals_over_level_2_threshold_concentration': [],
            'num_intervals_over_level_3_threshold_concentration': [],
            'num_intervals_over_level_4_threshold_concentration': [],
            'num_intervals_over_level_5_threshold_concentration': [],

            'datetime_utc': []
        }

        # Iterate over the dataframe using a rolling window
        for i in range(len(df) - num_30min_intervals):
            window = df.iloc[i:i + num_30min_intervals]['new_corrected']

           

            metrics['num_intervals_over_level_1_threshold_concentration'].append(window.gt(9.1).sum())
            metrics['num_intervals_over_level_2_threshold_concentration'].append(window.gt(35.5).sum())
            metrics['num_intervals_over_level_3_threshold_concentration'].append(window.gt(55.5).sum())
            metrics['num_intervals_over_level_4_threshold_concentration'].append(window.gt(125.5).sum())
            metrics['num_intervals_over_level_5_threshold_concentration'].append(window.gt(225.5).sum())

            metrics['datetime_utc'].append(df.index[i])


        # Convert dictionary of lists to DataFrame
        combined = pd.DataFrame(metrics)

        combined['Sensor Location'] = df['Sensor Location'].iloc[0]  # Assumes all entries have the same Sensor Location
        all_dfs_events.append(combined)

    return all_dfs_events

def Processing_rolling_window_dfs(original_dfs, dfs, observation_window, threshold_exceedance_time, lull_time):
    """
    Processes rolling window data and generates consolidated results based on event detection parameters.

    Parameters:
        original_dfs (list): Original sensor DataFrames.
        dfs (list): DataFrames for rolling window data.
        observation_window (int): Length of an observation window in hours.
        threshold_exceedance_time (float): Threshold time in hours for event detection.
        lull_time (float): Time in hours between events to combine them.

    Returns:
        dict: Consolidated results with events organized by varying levels.
    """
    consolidated_results = {f'level_{i}': [] for i in range(1, 6)}

    # Minimum number of intervals to be considered an event
    min_num_instances_to_be_an_event = int(threshold_exceedance_time * 2)

    for index, df in enumerate(dfs):
        combined = df.copy()

        for level in range(5, 0, -1):
            level_key = f'num_intervals_over_level_{level}_threshold_concentration'

            # Filter the DataFrame to include rows where event count meets the threshold
            filtered_combined = combined[(combined[level_key] / (threshold_exceedance_time * 2)) >= 1]

            if not filtered_combined.empty:
                events = []

                # Initialize the first event
                current_event = {
                    'Start Time': filtered_combined.iloc[0]['datetime_utc'],
                    'End Time': filtered_combined.iloc[0]['datetime_utc'] + timedelta(hours=observation_window),
                    'Max PM2.5': 0.0,
                    'Sensor Location': filtered_combined.iloc[0]['Sensor Location'],
                    'Average Concentration': None,
                    'Median Concentration': None
                }

                # Iterate through the filtered rows to detect and combine events
                for i in range(1, len(filtered_combined)):
                    current_time = filtered_combined.iloc[i]['datetime_utc']
                    previous_time = filtered_combined.iloc[i - 1]['datetime_utc']

                    if current_time - previous_time <= timedelta(hours=observation_window + lull_time):
                        # Extend the current event
                        current_event['End Time'] = current_time + timedelta(hours=observation_window)
                    else:
                        # Calculate average and median concentrations for the current event
                        avg_concentration, median_concentration, max_concentration = get_concentration_stats(
                            current_event['Start Time'], current_event['End Time'], original_dfs[index]
                        )

                        current_event['Average Concentration'] = avg_concentration
                        current_event['Median Concentration'] = median_concentration
                        current_event['Max PM2.5'] = max_concentration

                        # Save the current event
                        events.append(current_event)

                        # Initialize a new event
                        current_event = {
                            'Start Time': current_time,
                            'End Time': current_time + timedelta(hours=observation_window),
                            'Max PM2.5': 0.0,
                            'Sensor Location': filtered_combined.iloc[i]['Sensor Location'],
                            'Average Concentration': None,
                            'Median Concentration': None
                        }

                # Calculate average and median concentrations for the last event
                avg_concentration, median_concentration, max_concentration = get_concentration_stats(
                    current_event['Start Time'], current_event['End Time'], original_dfs[index]
                )
                current_event['Average Concentration'] = avg_concentration
                current_event['Median Concentration'] = median_concentration
                current_event['Max PM2.5'] = max_concentration
                events.append(current_event)

            else:
                # Create an empty DataFrame if no events are found
                events = pd.DataFrame(columns=[
                    'Start Time', 'End Time', 'Max PM2.5', 'Sensor Location', 'Average Concentration', 'Median Concentration'
                ])

            # Create a DataFrame from events
            events_df = pd.DataFrame(events)

            # Append the events DataFrame to the corresponding level in consolidated_results
            consolidated_results[f'level_{level}'].append(events_df)

    return consolidated_results




