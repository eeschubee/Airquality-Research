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
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Load and prepare the data
    df_main = pd.read_csv('input/outdoor_2021.csv')


    df_main['datetime_utc'] = pd.to_datetime(df_main['datetime_utc'])



    # Create a dictionary of DataFrames by location label
    df_dict = {label: df_main[df_main['location_label'] == label].copy() for label in df_main['location_label'].unique()}

    # Select specific DataFrames for processing
    sensor_dfs = [
        df_dict['SAFE_Happy_Camp_Community_Center'],
        df_dict['SAFE_Swillup_Creek'],
        df_dict['SAFE_Sandy_Bar_Creek'],
        df_dict['SAFE_Somes_Bar'],
        df_dict['KDNR_Outdoor'],
        df_dict['Butler_Creek'],
        df_dict['Forks_Of_Salmon'],
        df_dict['CARB_Cecilville'],
        df_dict['CARB_Sawyers_Bar']
    ]

    sensor_names = [
        "SAFE_Happy_Camp_Community_Center",
        "SAFE_Swillup_Creek",
        "SAFE_Sandy_Bar_Creek",
        "SAFE_Somes_Bar",
        "Orleans_KDNR_Outdoor",
        "Butler_Creek",
        "Forks_Of_Salmon",
        "CARB_Cecilville",
        "CARB_Sawyers_Bar"
    ]


    sensor_nicknames = {
        "SAFE_Happy_Camp_Community_Center": "HappyCamp",
        "SAFE_Swillup_Creek": "SwillupCreek",
        "SAFE_Sandy_Bar_Creek": "SandyBar",
        "SAFE_Somes_Bar": "SomesBar",
        "Orleans_KDNR_Outdoor": "Orleans",
        "Butler_Creek": "ButlerCreek",
        "Forks_Of_Salmon": "ForksSalmon",
        "CARB_Cecilville": "Cecilville",
        "CARB_Sawyers_Bar": "SawyersBar"
    }

    name_label = sensor_names

    # Define date range of interest
    date_range = pd.date_range(start='2021-07-01', end='2021-12-31', freq='15min')

    # Normalize DataFrames by setting index and reindexing
    for df in sensor_dfs:
        df.set_index('datetime_utc', inplace=True)
        df = df.reindex(date_range).copy()


    # Apply deviation calculations to each DataFrame
    for df in sensor_dfs:
        calculate_deviations(df)

    # Resample DataFrames to 30-minute intervals and add 'Sensor Location'
    resampled_dfs = []

    for df, name in zip(sensor_dfs, sensor_names):
        df_resampled = df['new_corrected'].resample('30min').mean().reset_index()
        df_resampled['Sensor Location'] = name
        resampled_dfs.append(df_resampled)



    # Filter DataFrames by the date range
    new_dfs = [
        df[(df['datetime_utc'] >= '2021-08-01') & (df['datetime_utc'] <= '2021-12-31')].copy()
        for df in resampled_dfs
    ]


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
        output_directory = f"output_3/{varying_parameter}/range_comprison_{params['observation window']}H_{params['threshold exceedance time']}H_{params['lull time']}H"

        Scatter_Event_Graph_AllLevels_PDF_rolling_window(
            events_dfs['level_5'], events_dfs['level_4'], events_dfs['level_3'], events_dfs['level_2'], events_dfs['level_1'],
            new_dfs, name_label, output_directory, [ params['observation window'],
            params['threshold exceedance time'], params['lull time'] ], sensor_nicknames
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
        output_directory=f"output_3/{varying_parameter}/Bar_Graphs",          # Directory for saving files
        varying_suffixes=varying_suffixes,          # Mapping of indices to suffix names for the varying parameter
        legend_labels=legend_labels,             # Custom legend labels
        varying_param_name = varying_parameter,
        constant_parameters = constant_parameters         # Name of the varying parameter (e.g., 'Threshold', 'Lull Time', 'Observation Window')
    ) 







if __name__ == "__main__":
    main()
