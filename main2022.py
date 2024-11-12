# main.py
import os                                    # For creating directories
import pandas as pd                          # For handling DataFrames
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For plotting
import matplotlib as mpl                     # For resetting plot configurations
from matplotlib.backends.backend_pdf import PdfPages  # For saving plots in a PDF
from datetime import timedelta               # For handling time intervals
from matplotlib import font_manager

# Import custom functions from your other modules
from data_processing_concise import (
    calculate_deviations,
    rolling_window_event_data,
    Processing_rolling_window_dfs,
    generate_parameter_list
    
)
from pdf_generation import (
    Scatter_Event_Graph_AllLevels_PDF_rolling_window,
    save_event_info_as_csv_5,
    generate_event_graphs_dynamic,
    process_event_data
    
)


def main():

    def save_event_info_as_csv_2022(consolidated_results, original_dfs, names, output_dir):
        """
        Save event information and statistics to a CSV file, ensuring required columns are present.

        Parameters:
            consolidated_results: dict of event DataFrames by levels (e.g., 'level_1', 'level_2', etc.).
            original_dfs: list of original DataFrames for reference.
            names: list of location names.
            output_dir: directory to save the CSV file.

        Returns:
            general_event_data_per_location: List of DataFrames with summary statistics for each location.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        levels = {'level_5': 5, 'level_4': 4, 'level_3': 3, 'level_2': 2, 'level_1': 1}
        output_file_path = os.path.join(output_dir, "all_events_summary_5.csv")

        general_event_data_per_location = []

        with open(output_file_path, 'w') as file:
            for i, name in enumerate(names):
                combined_df = pd.DataFrame()
                level_data = {level: [] for level in levels}
                event_stats = {level: {'count': 0, 'total_days': 0} for level in levels}

                for level_key, level_value in levels.items():
                    # Ensure the level exists in consolidated_results and the index is within bounds
                    if level_key not in consolidated_results or i >= len(consolidated_results[level_key]):
                        continue

                    df = consolidated_results[level_key][i]

                    # Skip empty DataFrames
                    if df.empty:
                        continue

                    # Verify required columns in the DataFrame
                    required_columns = ['Start Time', 'End Time', 'new_corrected']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Warning: Missing columns {missing_columns} in {name}, level {level_key}. Skipping...")
                        continue

                    # Clean and process data
                    df_cleaned = calculate_event_concentrations(df, original_dfs[i])
                    df_cleaned['Event Level'] = f'Level {level_value}'
                    df_cleaned['End Time'] = pd.to_datetime(df_cleaned['End Time'])
                    df_cleaned['Start Time'] = pd.to_datetime(df_cleaned['Start Time'])

                    df_cleaned['Event Duration (days)'] = (
                        (df_cleaned['End Time'] - df_cleaned['Start Time']).dt.total_seconds() / 86400
                    )
                    combined_df = pd.concat([combined_df, df_cleaned], ignore_index=True)

                    # Aggregate statistics and collect data
                    for _, event in df_cleaned.iterrows():
                        mask = (original_dfs[i]['datetime_utc'] >= event['Start Time']) & (
                            original_dfs[i]['datetime_utc'] <= event['End Time']
                        )
                        level_data[level_key].extend(original_dfs[i].loc[mask, 'new_corrected'].tolist())
                        event_stats[level_key]['count'] += 1
                        event_stats[level_key]['total_days'] += (
                            (event['End Time'] - event['Start Time']).total_seconds() / 86400
                        )

                # Sort by 'Start Time' and 'Event Level' if columns exist
                if 'Start Time' in combined_df.columns and 'Event Level' in combined_df.columns:
                    combined_df = combined_df.sort_values(by=['Start Time', 'Event Level'])
                else:
                    print(f"Warning: 'Start Time' or 'Event Level' missing in combined DataFrame for {name}.")

                # Write combined event information
                file.write(f"{name}\nEvent Information:\n")
                if not combined_df.empty:
                    file.write(combined_df.to_csv(index=False))
                file.write("\n")

                # Calculate and write statistics
                stats_data = []
                for level_key, data in level_data.items():
                    median_concentration = np.median(data) if data else 0
                    average_concentration = np.mean(data) if data else 0
                    ave_event_duration = (
                        event_stats[level_key]['total_days'] / event_stats[level_key]['count']
                        if event_stats[level_key]['count'] > 0 else 0
                    )
                    stats_data.append([
                        level_key.capitalize(),
                        median_concentration,
                        average_concentration,
                        event_stats[level_key]['count'],
                        event_stats[level_key]['total_days'],
                        ave_event_duration
                    ])

                stats_columns = [
                    'Event Level', 'Median Concentration', 'Average Concentration',
                    'Total Event Count', 'Total Event Days', 'Average Event Duration'
                ]
                stats_df = pd.DataFrame(stats_data, columns=stats_columns)
                file.write(stats_df.to_csv(index=False))
                file.write("\n\n")
                general_event_data_per_location.append(stats_df)

        return general_event_data_per_location




    sensor_files = {
    'SAFE_Happy_Camp_Community_Center': 'input/Happy_Camp_2022_v1.csv',
    'SAFE_Somes_Bar': 'input/Somes_Bar_2022_v1.csv',
    'Orleans_KDNR_Outdoor': 'input/Orleans_MKWC_2022_v1.csv',
    'Butler_Creek': 'input/Butler_Creek_2022_v1.csv',
    'Forks_Of_Salmon': 'input/Forks_of_Salmon_2022_v1.csv',
    'CARB_Cecilville': 'input/Cecilville_2022_v1.csv',
    'CARB_Sawyers_Bar': 'input/Sawyers_Bar_2022_v1.csv'
    }

    sensor_dfs = []  # List to hold processed DataFrames
    sensor_names = list(sensor_files.keys())  # Corresponding sensor names
    sensor_nicknames = {
        "SAFE_Happy_Camp_Community_Center": "HappyCamp",
        "SAFE_Somes_Bar": "SomesBar",
        "Orleans_KDNR_Outdoor": "Orleans",
        "Butler_Creek": "ButlerCreek",
        "Forks_Of_Salmon": "ForksSalmon",
        "CARB_Cecilville": "Cecilville",
        "CARB_Sawyers_Bar": "SawyersBar"
    }

    name_label = sensor_names

    # Process each file
    for name, path in sensor_files.items():
        try:
            # Load DataFrame
            df = pd.read_csv(path)

            # Ensure the time column exists and is renamed
            if 'time_stamp' in df.columns:
                df.rename(columns={'time_stamp': 'datetime_utc'}, inplace=True)
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc']).dt.tz_localize(None)
            else:
                raise KeyError(f"'time_stamp' column missing in {path}.")

            # Ensure the corrected column exists and is renamed
            if 'corrected' in df.columns:
                df.rename(columns={'corrected': 'new_corrected'}, inplace=True)
            else:
                raise KeyError(f"'corrected' column missing in {path}.")

            # Set 'datetime_utc' as index
            df.set_index('datetime_utc', inplace=True)

            # Reindex to ensure uniform time intervals
            date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='15min')
            df = df.reindex(date_range).copy()

            # Append to sensor_dfs
            sensor_dfs.append(df)
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Resample DataFrames to 30-minute intervals and add 'Sensor Location'
    resampled_dfs = []
    for df, name in zip(sensor_dfs, sensor_names):
        df_resampled = df['new_corrected'].resample('30min').mean().reset_index()
        df_resampled.rename(columns={'index': 'datetime_utc'}, inplace=True)  # Ensure datetime column is named
        df_resampled['Sensor Location'] = name
        resampled_dfs.append(df_resampled)

    # Filter DataFrames by the date range (post-resampling)
    new_dfs = [
        df[(df['datetime_utc'] >= '2022-01-01') & (df['datetime_utc'] <= '2022-12-31')].copy()
        for df in resampled_dfs
    ]

    # For verification
    for df in new_dfs:
        print(df.head())

    '''
    constant_parameters = {'threshold exceedance time': 3,'observation window' : 24}
    varying_parameter = 'lull time'
    varying_values = [1, 3, 6,12]
    '''

    

    constant_parameters = {'lull time': 3,'observation window' : 24}
    varying_parameter = 'threshold exceedance time'
    varying_values = [1.5, 3, 6,12,18]

    '''
    constant_parameters = {'threshold exceedance time': 3,'lull time': 3}
    varying_parameter = 'observation window'
    varying_values = [12, 24, 36]


    '''

    param_unit= 'H'


    

    Event_detection_parameters_list = generate_parameter_list(varying_parameter, varying_values, constant_parameters)

    # Generate Sufixes
    varying_suffixes = {index: f"{value}{param_unit}" for index, value in enumerate(varying_values)}

    # Generate the legend_labels list
    legend_labels = [f"{value}{param_unit}" for value in varying_values]






    # Process rolling window data
    event_detection_parameter_data = []
    for params in Event_detection_parameters_list:

        rolling_data = rolling_window_event_data(new_dfs, params['observation window'])


        events_dfs = Processing_rolling_window_dfs(
            new_dfs, rolling_data, params['observation window'],
            params['threshold exceedance time'], params['lull time']
        )
        event_detection_parameter_data.append(events_dfs)




    # Generate event range comparison results
    Event_Range_Comparison = []
    for i, events_dfs in enumerate(event_detection_parameter_data):
        params = Event_detection_parameters_list[i]
        output_directory = f"output_2022/{varying_parameter}/range_comprison_{params['observation window']}H_{params['threshold exceedance time']}H_{params['lull time']}H"

        Scatter_Event_Graph_AllLevels_PDF_rolling_window(
            events_dfs['level_5'], events_dfs['level_4'], events_dfs['level_3'], events_dfs['level_2'], events_dfs['level_1'],
            new_dfs, name_label, output_directory, [ params['observation window'],
            params['threshold exceedance time'], params['lull time'] ], sensor_nicknames,is_2022 = True
        )

        event_summary = save_event_info_as_csv_5(events_dfs, new_dfs, name_label, output_directory)
        Event_Range_Comparison.append(event_summary)




    # Process all event data
    all_locations_event_detection_params_data = process_event_data(
        Event_Range_Comparison=Event_Range_Comparison
    )




    generate_event_graphs_dynamic(
        location_event_data=all_locations_event_detection_params_data,
        location_labels=name_label,
        nick_names = sensor_nicknames,
        output_directory=f"output_2022/{varying_parameter}/Bar_Graphs",          # Directory for saving files
        varying_suffixes=varying_suffixes,          # Mapping of indices to suffix names for the varying parameter
        legend_labels=legend_labels,             # Custom legend labels
        varying_param_name = varying_parameter,
        constant_parameters = constant_parameters         # Name of the varying parameter (e.g., 'Threshold', 'Lull Time', 'Observation Window')
    ) 







if __name__ == "__main__":
    main()
