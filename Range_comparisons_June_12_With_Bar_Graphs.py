import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
# for creating
import os
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.lines import Line2D

from matplotlib.animation import FuncAnimation

mpl.rcParams.update(mpl.rcParamsDefault)

#Specify file with location
df_main = pd.read_csv('outdoor_2021.csv')

df_main['datetime_utc'] = pd.to_datetime(df_main['datetime_utc'])

# Identify unique location (sensor) labels
temp = df_main.location_label.unique()
#print(temp)


#create a dictionary using unique location labels. This will be used to split the data frame
df_dict = {sale_v: df_main[df_main['location_label'] == sale_v] for sale_v in df_main.location_label.unique()}

df_butler = df_dict['Butler_Creek']
df_cecilville = df_dict['CARB_Cecilville']
df_happy_camp_cc = df_dict['SAFE_Happy_Camp_Community_Center']
df_sawyer = df_dict['CARB_Sawyers_Bar']
df_forks = df_dict['Forks_Of_Salmon']
df_kdnr_out = df_dict['KDNR_Outdoor']
df_somesbar = df_dict['SAFE_Somes_Bar']
df_sandybar = df_dict['SAFE_Sandy_Bar_Creek']
df_swillup = df_dict['SAFE_Swillup_Creek']


#Create sensor location names for later use in graphs
sensor_location_names = ["SAFE_Happy_Camp_Community_Center", "SAFE_Swillup_Creek", "SAFE_Sandy_Bar_Creek", "SAFE_Somes_Bar", "Orleans_KDNR_Outdoor", "Butler_Creek", "Forks_Of_Salmon", "CARB_Cecilville", "CARB_Sawyers_Bar"]


#create a list of the needed data frames from the ones selected
sensor_dfs = (df_happy_camp_cc, df_swillup, df_sandybar, df_somesbar, df_kdnr_out, df_butler, df_forks, df_cecilville, df_sawyer)


#defining the function for subtracting dates

def get_difference(startdate, enddate):
    diff = enddate - startdate
    return diff.days

#initializing dates
# this was the first and last date of the data set. -- hard coded


#the output from days*24*4 is used to create a time series that starts juy 1 and is every 15 min
#reindex - generate 15 minute interval time index

sensor_list_df =  (df_happy_camp_cc, df_swillup, df_sandybar, df_somesbar, df_kdnr_out, df_butler, df_forks, df_cecilville, df_sawyer)

date_index2 = pd.date_range('2021/07/01', periods=17664, freq='15min')



#### THESE ARE THE ONES ACTUALLY BEING USED

####

sensor_list_gf = (df_happy_camp_cc, df_swillup, df_sandybar, df_somesbar, df_kdnr_out, df_butler, df_forks, df_cecilville, df_sawyer)

name_label = ["SAFE_Happy_Camp_Community_Center", "SAFE_Swillup_Creek", "SAFE_Sandy_Bar_Creek", "SAFE_Somes_Bar", "Orleans_KDNR_Outdoor", "Butler_Creek", "Forks_Of_Salmon", "CARB_Cecilville", "CARB_Sawyers_Bar"]


####

#the data ranges were intrested in

min_date = pd.to_datetime('2021-07-01')
max_date = pd.to_datetime('2021-12-31')
date_index = pd.date_range(start = min_date, end = max_date, freq='15min')


for df in sensor_list_gf:

    is_timezone_aware = df['datetime_utc'].dt.tz is not None
    #print(is_timezone_aware)



    # Editing Original DF to index by date_time UTC

    #Setting inplace=True modifies the original DataFrame, meaning no new DataFrame is created and the changes are applied directly to df.

    #Setting append=False means that the new index ("datetime_utc") will replace the existing index.

    #Setting drop=True means that after "datetime_utc" is set as the index, it will no longer appear as a column in the DataFrame.

    df.set_index(["datetime_utc"], inplace = True,append = False, drop = True)
    #print(f"before {len(df)}")
    #this reindexes date_index2 to fill in misssing data points.
    df = df.reindex(date_index)

    # this is a check to see how many of the 15 min intervals have N/A data.
    checkna = df['longitude'].isna()

    #print(f"after {len(df)}")
    #print(df.head())

## USE THESE FUNCTIONS to verify the values in DF are correct 
def compute_absolute_deviation(df, reference_column, comparison_column):
    return np.abs(df[reference_column] - df[comparison_column])

def compute_fraction_deviation(df, reference_column, comparison_column):
    return np.abs(df[reference_column] - df[comparison_column]) / df[reference_column]

# Check collumn
for df in sensor_list_df:
   #these colums already exsisted for 2021 because it was pre cleaned 2022 dose this manually with a data cleaning code.
    # Add the 'check' column based on the conditions

    # Compute absolute and fraction deviations
    df['ab_deviation_absolute'] = compute_absolute_deviation(df, 'pm25_A', 'pm25_B')
    df['ab_deviation_fraction'] = compute_fraction_deviation(df, 'pm25_A', 'pm25_B')


    df['check'] = (df['ab_deviation_absolute'] > 5.0) & (df['ab_deviation_fraction'] > 0.7)

    # Set 'pm25_avg' to NaN where 'check' is True
    df.loc[df['check'], 'pm25_avg'] = np.nan
    df.loc[~df['check'], 'pm25_avg'] = (df['pm25_A'] + df['pm25_B']) / 2

    df.loc[df['check'], 'new_corrected'] = np.nan
    df.loc[~df['check'], 'new_corrected'] = 0.79 * df['pm25_avg'] - 7.96 

    df.loc[df['new_corrected']<0,'new_corrected'] = 0

    #print(df.head(50))




   # createsa  col in df called check that either T/F


# time difference function
'''
def convert_timedelta(duration):
   days, seconds = duration.days, duration.seconds
   hours = days * 24 + seconds // 3600
   minutes = (seconds % 3600) // 60
   seconds = (seconds % 60)
   return hours, minutes, seconds
'''

COLORS = ['purple','brown','violet','slategrey','khaki',
                'gray','silver','rosybrown','firebrick','darksalmon','sienna','sandybrown',
                'olivedrab','chartreuse','palegreen','darkgreen','seagreen','navy','peachpuff','darkorange',
                'navajowhite','lemonchiffon','mediumseagreen','cadetblue','skyblue','dodgerblue','slategray']

resampled_dfs = []

# Iterate over sensor_list_gf and use 30min averages.

for k, df in enumerate(sensor_list_gf):

    df_resampled = df['new_corrected'].resample('30min').mean().reset_index()

    # Add a column for sensor location
    df_resampled['Sensor Location'] = name_label[k]
    #print(df_resampled.head())

    # Append the resampled DataFrame to the list
    resampled_dfs.append(df_resampled)



new_dfs = []
# Iterate over each DataFrame in resampled_df and normalize the end and start date and verify all data points are in that range

# Edit this to be able to take in custom date ranges.
for i, df in enumerate(resampled_dfs):

    # Find the minimum and maximum dates in the 'datetime_utc' column
    #min_date = df['datetime_utc'].min().normalize()  # normalize to midnight
    #max_date = df['datetime_utc'].max().normalize()  # normalize to midnight

    min_date = pd.to_datetime('2021-07-01')
    max_date = pd.to_datetime('2021-12-31')

    #Filter the DataFrame based on the min and max dates
    filtered_df = df[(df['datetime_utc'] >= min_date ) & (df['datetime_utc'] <= max_date)]


    # Append the filtered DataFrame to the list
    new_dfs.append(filtered_df)
    #new_dfs.append(df)

def rolling_window_event_data(dfs, event_length_in_days):
    
    all_dfs_events = []
    num_30min_intervals = int(event_length_in_days * 48)
    

    for df in dfs:
        # Ensure 'datetime_utc' is a datetime type for proper processing
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df.set_index('datetime_utc')

        # Prepare a dictionary to collect all metrics
        metrics = {
            'Max PM2.5': [],
            'Average PM2.5': [],
            'NaN Count': [],
            'Below Zero Count': [],
            '0-55.5 Count': [],
            '55.6-150.5 Count': [],
            '150.6-250.5 Count': [],
            '250.6-700 Count': [],
            'cum level 3 event count': [],
            'cum level 2 event count': [],
            'cum level 1 event count': [],
            'datetime_utc': []
        }

        # Iterate over the dataframe using a rolling window
        for i in range(len(df) - num_30min_intervals):
            window = df.iloc[i:i + num_30min_intervals]['new_corrected']

            metrics['Max PM2.5'].append(window.max())
            metrics['Average PM2.5'].append(window.mean())
            metrics['NaN Count'].append(window.isna().sum())
            metrics['Below Zero Count'].append((window < 0).sum())
            metrics['0-55.5 Count'].append(window.between(0, 55.5).sum())
            metrics['55.6-150.5 Count'].append(window.between(55.6, 150.5).sum())
            metrics['150.6-250.5 Count'].append(window.between(150.6, 250.5).sum())
            metrics['250.6-700 Count'].append(window.between(250.6, 700).sum())
            metrics['cum level 3 event count'].append(window.gt(55.5).sum())
            metrics['cum level 2 event count'].append(window.gt(150.5).sum())
            metrics['cum level 1 event count'].append(window.gt(250.5).sum())
            metrics['datetime_utc'].append(df.index[i])

        # Convert dictionary of lists to DataFrame
        combined = pd.DataFrame(metrics)
        combined['Sensor Location'] = df['Sensor Location'].iloc[0]  # Assumes all entries have the same Sensor Location
        all_dfs_events.append(combined)

    return all_dfs_events


def Processing_rolling_window_dfs(original_dfs,dfs,event_length_in_days, hours_over_threshold_to_be_considered_an_event, hours_between_events):

    consolidated_results = {'level_3': [], 'level_2': [], 'level_1': []}

    num_30min_intervals = int(event_length_in_days * 48)
    min_num_instances_to_be_an_event = int(hours_over_threshold_to_be_considered_an_event * 2)

    for index, df in enumerate(dfs):
        combined = df.copy()

        for level in ['cum level 3 event count', 'cum level 2 event count', 'cum level 1 event count']:
            # Filter the dataframe to only include rows where the event count is greater than 1

            filtered_combined = combined[(combined[level] / (hours_over_threshold_to_be_considered_an_event * 2))> 1]

            if not filtered_combined.empty:
                events = []
                current_event = {
                    'Start Time': filtered_combined.iloc[0]['datetime_utc'],
                    'End Time': filtered_combined.iloc[0]['datetime_utc'] + timedelta(days=event_length_in_days),
                    'Max PM2.5': filtered_combined.iloc[0]['Max PM2.5'],
                    'Average PM2.5': filtered_combined.iloc[0]['Average PM2.5'],
                    'Sensor Location': filtered_combined.iloc[0]['Sensor Location'],
                    'Average Concentration': 0.0,
                    'Median Concentration': 0.0
                }

                for i in range(1, len(filtered_combined)):
                    current_time = filtered_combined.iloc[i]['datetime_utc']
                    previous_time = filtered_combined.iloc[i - 1]['datetime_utc']

                    if current_time - previous_time <= timedelta(hours=event_length_in_days*24+hours_between_events):
                        # Extend the current event
                        current_event['End Time'] = current_time + timedelta(days=event_length_in_days)
                        current_event['Max PM2.5'] = max(current_event['Max PM2.5'], filtered_combined.iloc[i]['Max PM2.5'])
                        current_event['Average PM2.5'] = (current_event['Average PM2.5'] * (i) + filtered_combined.iloc[i]['Average PM2.5']) / (i + 1)
                        
                    else:
                        # Calculate average and median concentrations for the current event
                        avg_concentration, median_concentration = get_concentration_stats(current_event['Start Time'], current_event['End Time'], original_dfs[index])
                        current_event['Average Concentration'] = avg_concentration
                        current_event['Median Concentration'] = median_concentration

                        # Save the current event and start a new one
                        events.append(current_event)
                        current_event = {
                            'Start Time': current_time,
                            'End Time': current_time + timedelta(days=event_length_in_days),
                            'Max PM2.5': filtered_combined.iloc[i]['Max PM2.5'],
                            'Average PM2.5': filtered_combined.iloc[0]['Average PM2.5'],
                            'Sensor Location': filtered_combined.iloc[i]['Sensor Location'],
                            'Average Concentration': 0.0,
                            'Median Concentration': 0.0
                        }
                # Calculate average and median concentrations for the last event
                avg_concentration, median_concentration = get_concentration_stats(current_event['Start Time'], current_event['End Time'], original_dfs[index])
                current_event['Average Concentration'] = avg_concentration
                current_event['Median Concentration'] = median_concentration
                events.append(current_event)
            else:
                # Create an empty DataFrame if no events are found
                events = pd.DataFrame(columns=['Start Time', 'End Time', 'Max PM2.5', 'Average PM2.5', 'Sensor Location', 'Average Concentration', 'Median Concentration'])

            # Create DataFrame from events
            events_df = pd.DataFrame(events)

            # Append the events dataframe to the corresponding level in consolidated_results
            if level == 'cum level 3 event count':
                consolidated_results['level_3'].append(events_df)
            elif level == 'cum level 2 event count':
                consolidated_results['level_2'].append(events_df)
            elif level == 'cum level 1 event count':
                consolidated_results['level_1'].append(events_df)


    return consolidated_results







def rolling_window_event_analysis_all_3_levels_with_median_and_avg(dfs, event_length_in_days, hours_over_threshold_to_be_considered_an_event):

    consolidated_results = {'level_3': [], 'level_2': [], 'level_1': []}

    num_30min_intervals = int(event_length_in_days * 48)
    min_num_instances_to_be_an_event = int(hours_over_threshold_to_be_considered_an_event * 2)

    for df in dfs:
        # Ensure 'datetime_utc' is a datetime type for proper processing
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df.set_index('datetime_utc')

        # Prepare a dictionary to collect all metrics
        metrics = {
            'Max PM2.5': [],
            'Average PM2.5': [],
            'NaN Count': [],
            'Below Zero Count': [],
            '0-55.5 Count': [],
            '55.6-150.5 Count': [],
            '150.6-250.5 Count': [],
            '250.6-700 Count': [],
            'cum level 3 event count': [],
            'cum level 2 event count': [],
            'cum level 1 event count': [],
            'datetime_utc': []
        }

        # Iterate over the dataframe using a rolling window
        for i in range(len(df) - num_30min_intervals):
            window = df.iloc[i:i + num_30min_intervals]['new_corrected']

            metrics['Max PM2.5'].append(window.max())
            metrics['Average PM2.5'].append(window.mean())
            metrics['NaN Count'].append(window.isna().sum())
            metrics['Below Zero Count'].append((window < 0).sum())
            metrics['0-55.5 Count'].append(window.between(0, 55.5).sum())
            metrics['55.6-150.5 Count'].append(window.between(55.6, 150.5).sum())
            metrics['150.6-250.5 Count'].append(window.between(150.6, 250.5).sum())
            metrics['250.6-700 Count'].append(window.between(250.6, 700).sum())
            metrics['cum level 3 event count'].append((window.gt(55.5).sum() / min_num_instances_to_be_an_event))
            metrics['cum level 2 event count'].append((window.gt(150.5).sum() / min_num_instances_to_be_an_event))
            metrics['cum level 1 event count'].append((window.gt(250.5).sum() / min_num_instances_to_be_an_event))
            metrics['datetime_utc'].append(df.index[i])

        # Convert dictionary of lists to DataFrame
        combined = pd.DataFrame(metrics)
        combined['Sensor Location'] = df['Sensor Location'].iloc[0]  # Assumes all entries have the same Sensor Location


        for level in ['cum level 3 event count', 'cum level 2 event count', 'cum level 1 event count']:
            # Filter the dataframe to only include rows where the event count is greater than 1
            filtered_combined = combined[combined[level] > 1]

            if not filtered_combined.empty:
                events = []
                current_event = {
                    'Start Time': filtered_combined.iloc[0]['datetime_utc'],
                    'End Time': filtered_combined.iloc[0]['datetime_utc'] + timedelta(days=event_length_in_days),
                    'Max PM2.5': filtered_combined.iloc[0]['Max PM2.5'],
                    'Average PM2.5': filtered_combined.iloc[0]['Average PM2.5'],
                    'Sensor Location': filtered_combined.iloc[0]['Sensor Location'],
                    'Average Concentration': 0.0,
                    'Median Concentration': 0.0
                }

                for i in range(1, len(filtered_combined)):
                    current_time = filtered_combined.iloc[i]['datetime_utc']
                    previous_time = filtered_combined.iloc[i - 1]['datetime_utc']

                    if current_time - previous_time <= timedelta(hours=event_length_in_days*24*1.083):
                        # Extend the current event
                        current_event['End Time'] = current_time + timedelta(days=event_length_in_days)
                        current_event['Max PM2.5'] = max(current_event['Max PM2.5'], filtered_combined.iloc[i]['Max PM2.5'])
                        current_event['Average PM2.5'] = (current_event['Average PM2.5'] * (i) + filtered_combined.iloc[i]['Average PM2.5']) / (i + 1)
                    else:
                        # Calculate average and median concentrations for the current event
                        avg_concentration, median_concentration = get_concentration_stats(current_event['Start Time'], current_event['End Time'], df)
                        current_event['Average Concentration'] = avg_concentration
                        current_event['Median Concentration'] = median_concentration

                        # Save the current event and start a new one
                        events.append(current_event)
                        current_event = {
                            'Start Time': current_time,
                            'End Time': current_time + timedelta(days=event_length_in_days),
                            'Max PM2.5': filtered_combined.iloc[i]['Max PM2.5'],
                            'Average PM2.5': filtered_combined.iloc[i]['Average PM2.5'],
                            'Sensor Location': filtered_combined.iloc[i]['Sensor Location'],
                            'Average Concentration': 0.0,
                            'Median Concentration': 0.0
                        }
                # Calculate average and median concentrations for the last event
                avg_concentration, median_concentration = get_concentration_stats(current_event['Start Time'], current_event['End Time'], df)
                current_event['Average Concentration'] = avg_concentration
                current_event['Median Concentration'] = median_concentration
                events.append(current_event)
            else:
                # Create an empty DataFrame if no events are found
                events = pd.DataFrame(columns=['Start Time', 'End Time', 'Max PM2.5', 'Average PM2.5', 'Sensor Location', 'Average Concentration', 'Median Concentration'])

            # Create DataFrame from events
            events_df = pd.DataFrame(events)

            # Append the events dataframe to the corresponding level in consolidated_results
            if level == 'cum level 3 event count':
                consolidated_results['level_3'].append(events_df)
            elif level == 'cum level 2 event count':
                consolidated_results['level_2'].append(events_df)
            elif level == 'cum level 1 event count':
                consolidated_results['level_1'].append(events_df)
    return consolidated_results



def get_concentration_stats(start_time, end_time, df):

    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df = df.set_index('datetime_utc')


    mask = (df.index >= start_time) & (df.index <= end_time)
    filtered_data = df.loc[mask, 'new_corrected']
    average_concentration = filtered_data.mean()
    median_concentration = filtered_data.median()
    return average_concentration, median_concentration





def Scatter_Event_Graph_AllLevels_PDF_rolling_window(level_3_events_detected, level_2_events_detected, level_1_events_detected, new_dfs, name_label, output_dir, percise_results_dfs, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    days = int(params[0] * 24)
    duration = params[1]
    min_time_between_events = params[2]


    levels_detected = [level_1_events_detected, level_2_events_detected, level_3_events_detected]
    level_keys = [1, 2, 3]
    level_colors = ['maroon', 'purple', 'red']

    overall_min_date = min(df['datetime_utc'].min() for df in new_dfs)
    overall_max_date = max(df['datetime_utc'].max() for df in new_dfs)

    # Setup PDF file to save plots
    File_Name = f'Duration{days}_Threshold{duration}_min{min_time_between_events}_locations_events.pdf'
    output_filepath = os.path.join(output_dir, File_Name)

    with PdfPages(output_filepath) as pdf:
        for i, df in enumerate(new_dfs):
            plt.figure(figsize=(14, 6))
            conditions = [
                (df['new_corrected'] >= 250.5),
                (df['new_corrected'] >= 150.5) & (df['new_corrected'] <= 250.4),
                (df['new_corrected'] >= 55.5) & (df['new_corrected'] <= 150.4),
                (df['new_corrected'] >= 35.5) & (df['new_corrected'] <= 55.4),
                (df['new_corrected'] >= 12.1) & (df['new_corrected'] <= 35.4),
                (df['new_corrected'] >= 0.0) & (df['new_corrected'] <= 12.0)
            ]
            colors = ['maroon', 'purple', 'red', 'orange', 'yellow', 'green']
            df.loc[:, 'color'] = np.select(conditions, colors, default='blue')

            plt.plot(percise_results_dfs[i]['datetime_utc'], percise_results_dfs[i]['Average PM2.5'], 2, color="black", alpha=1)

            for color in colors:
                subset = df[df['color'] == color]
                plt.scatter(subset['datetime_utc'], subset['new_corrected'], 1, color=color, alpha=.5)

            for level_idx, level_events in enumerate(levels_detected):
                if i < len(level_events) and level_events[i] is not None:
                    staticData = level_events[i]
                    for _, row in staticData.iterrows():
                        start_time = pd.to_datetime(row['Start Time'])
                        end_time = pd.to_datetime(row['End Time'])
                        if pd.isna(start_time) or pd.isna(end_time):
                            continue

                        highest_pm25 = row['Max PM2.5']
                        event_bar_heights = [float(highest_pm25), min(250.5, float(highest_pm25)), min(150.5, float(highest_pm25))]
                        event_bar_bottoms = [250.5, 150.5, 0]
                        bar_widths = [1, 1, 1]

                        plt.plot([start_time, start_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)
                        plt.plot([start_time, end_time], [event_bar_heights[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)
                        plt.plot([end_time, end_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)

            plt.title(f'PM2.5 Concentrations and Events for {name_label[i]} 2021')
            plt.xlabel('Time')
            plt.ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
            plt.xlim(overall_min_date, overall_max_date)
            plt.ylim(0, 800)
            plt.tight_layout()
            pdf.savefig()  # Save the current figure into the pdf
            plt.close()

    print(f"Saved all graphs into {output_filepath}")

def All_One_location_Scatter_Event_Graph_AllLevels_PDF_rolling_window(DFs_for_loc, new_dfs, name_label, output_dir, percise_results_dfs, INDEX, Param_list):


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #levels_detected = [level_1_events_detected, level_2_events_detected, level_3_events_detected]
    level_keys = [1, 2, 3]
    level_colors = ['maroon', 'purple', 'red']

    overall_min_date = min(df['datetime_utc'].min() for df in new_dfs)
    overall_max_date = max(df['datetime_utc'].max() for df in new_dfs)

    #overall_min_date = new_dfs[INDEX]['datetime_utc'].min()
    #overall_max_date = new_dfs[INDEX]['datetime_utc'].max()

    # Setup PDF file to save plots
    file_name = f"{name_label[INDEX]}_All_Param.pdf"
    output_filepath = os.path.join(output_dir, file_name)

    with PdfPages(output_filepath) as pdf:
        for i, df in enumerate(DFs_for_loc):
            plt.figure(figsize=(14, 6))
            conditions = [
                (new_dfs[INDEX]['new_corrected'] >= 250.5),
                (new_dfs[INDEX]['new_corrected'] >= 150.5) & (new_dfs[INDEX]['new_corrected'] <= 250.4),
                (new_dfs[INDEX]['new_corrected']>= 55.5) & (new_dfs[INDEX]['new_corrected'] <= 150.4),
                (new_dfs[INDEX]['new_corrected'] >= 35.5) & (new_dfs[INDEX]['new_corrected'] <= 55.4),
                (new_dfs[INDEX]['new_corrected'] >= 12.1) & (new_dfs[INDEX]['new_corrected'] <= 35.4),
                (new_dfs[INDEX]['new_corrected'] >= 0.0) & (new_dfs[INDEX]['new_corrected'] <= 12.0)
            ]
            colors = ['maroon', 'purple', 'red', 'orange', 'yellow', 'green']
            new_dfs[INDEX].loc[:, 'color'] = np.select(conditions, colors, default='blue')

            plt.plot(percise_results_dfs[INDEX]['datetime_utc'], percise_results_dfs[INDEX]['Average PM2.5'], 2, color="black", alpha=1)

            for color in colors:
                subset = new_dfs[INDEX][new_dfs[INDEX]['color'] == color]
                plt.scatter(subset['datetime_utc'], subset['new_corrected'], 1, color=color, alpha=.5)

            for level_idx, level_events in enumerate(df):

                if 0 < len(level_events):



                    staticData = level_events
                    for _, row in staticData.iterrows():

                        start_time = pd.to_datetime(row['Start Time'])

                        end_time = pd.to_datetime(row['End Time'])

                        if pd.isna(start_time) or pd.isna(end_time):
                            continue

                        highest_pm25 = row['Max PM2.5']
                        event_bar_heights = [float(highest_pm25), min(250.5, float(highest_pm25)), min(150.5, float(highest_pm25))]
                        event_bar_bottoms = [250.5, 150.5, 0]
                        bar_widths = [1, 1, 1]

                        plt.plot([start_time, start_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)
                        plt.plot([start_time, end_time], [event_bar_heights[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)
                        plt.plot([end_time, end_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=0.5)

            plt.title(f'{Param_list[i][0]} Days, {Param_list[i][1]} Hours over param ,{Param_list[i][0]} H betweenEvents, HPM2.5 Concentrations and Events for {name_label[INDEX]} 2021')
            plt.xlabel('Time')
            plt.ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
            plt.xlim(overall_min_date, overall_max_date)
            plt.ylim(0, 800)
            plt.tight_layout()
            pdf.savefig()  # Save the current figure into the pdf
            plt.close()

    print(f"Saved all graphs into {output_filepath}")

# 12 hour Processing.
def percision_df_processing(df):
    #print(len(df))
    #print(df.head())
    #print(df.tail())

    #Ensure 'datetime_utc' is a datetime type for proper resampling

    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

    # Resample the data by 24-hour intervals

    resampled = df.resample('12h', on='datetime_utc')['new_corrected']

    # Prepare a dictionary to collect all metrics

    metrics = {
        #'Min PM2.5': resampled.min(),
        'Max PM2.5': resampled.max(),

        'Average PM2.5': resampled.mean(),

        'NaN Count': resampled.apply(lambda x: x.isna().sum()),
        'Below Zero Count': resampled.apply(lambda x: (x < 0).sum()),

        '0-55.5 Count': resampled.apply(lambda x: x.between(0, 55.5).sum()),
        '55.6-150.5 Count': resampled.apply(lambda x: x.between(55.6, 150.5).sum()),
        '150.6-250.5 Count': resampled.apply(lambda x: x.between(150.6, 250.5).sum()),
        '250.6-700 Count': resampled.apply(lambda x: x.between(250.6, 700).sum()),

        #'Level 3 event count': resampled.apply(lambda x: (x.between(55.6, 150.5).sum() / 6) if x.between(55.6, 150.5).sum() > 6 else 0),
        #'Level 2 event count': resampled.apply(lambda x: (x.between(150.6, 250.5).sum()/6)if x.between(150.6, 250.5).sum() > 6 else 0),
        #'Level 1 event count': resampled.apply(lambda x: (x.gt(250.6).sum()/6)if x.gt(250.6).sum() > 6 else 0),

        'cum level 3 event count': resampled.apply(lambda x: (x.gt(55.5).sum()  / 6) ),
        'cum level 2 event count': resampled.apply(lambda x: (x.gt(150.5).sum() / 6) ),
        'cum level 1 event count': resampled.apply(lambda x: (x.gt(250.5).sum() / 6) )
    }

    # Convert dictionary of Series to DataFrame
    combined = pd.DataFrame(metrics)
    combined.reset_index(inplace=True)
    combined['Sensor Location'] = df['Sensor Location'].iloc[0]  # Assumes all entries have the same Sensor Location

    return combined



percise_results_dfs = []

for df in new_dfs:
    percise_results_df = percision_df_processing(df)
    percise_results_dfs.append(percise_results_df)
    print(percise_results_df.head(20))


#print(percise_results_dfs)


def calculate_event_concentrations(events_df, original_df):
    """Calculate average and median concentrations for each event."""
    for idx, row in events_df.iterrows():
        start_time = row['Start Time']
        end_time = row['End Time']
        mask = (original_df['datetime_utc'] >= start_time) & (original_df['datetime_utc'] <= end_time)
        filtered_data = original_df.loc[mask, 'new_corrected']
        events_df.at[idx, 'Average Concentration'] = filtered_data.mean()
        events_df.at[idx, 'Median Concentration'] = filtered_data.median()
    return events_df

def save_event_info_as_csv_5(consolidated_results, original_dfs, names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the levels and their numeric equivalents
    levels = {'level_3': 3, 'level_2': 2, 'level_1': 1}

    # Path for the output CSV file
    output_file_path = os.path.join(output_dir, "all_events_summary_5.csv")

    # Open the file for writing
    with open(output_file_path, 'w') as file:

        general_event_data_per_location = [] 

        for i, name in enumerate(names):
            combined_df = pd.DataFrame()
            level_data = {level: [] for level in levels}  # Store raw data for each level
            event_stats = {level: {'count': 0, 'total_days': 0} for level in levels}  # Event count and total days

            # Process each event level
            for level_key, level_value in levels.items():
                if i >= len(consolidated_results[level_key]):
                    continue
                df = consolidated_results[level_key][i]
                if df.empty:
                    continue

                # Clean and process the data
                df_cleaned = calculate_event_concentrations(df, original_dfs[i])
                df_cleaned['Event Level'] = f'Level {level_value}'
                combined_df = pd.concat([combined_df, df_cleaned], ignore_index=True)

                # Collect data and calculate stats for each event
                for _, event in df_cleaned.iterrows():
                    mask = (original_dfs[i]['datetime_utc'] >= event['Start Time']) & (original_dfs[i]['datetime_utc'] <= event['End Time'])
                    level_data[level_key].extend(original_dfs[i].loc[mask, 'new_corrected'].tolist())
                    event_stats[level_key]['count'] += 1
                    event_stats[level_key]['total_days'] += (event['End Time'] - event['Start Time']).total_seconds() / 86400  # Convert seconds to days

            # Sort by Start Time and Event Level
            sorted_combined = combined_df.sort_values(by=['Start Time', 'Event Level'])

            # Write event information
            file.write(f"{name}\nEvent Information:\n")
            file.write(sorted_combined.to_csv(index=False))
            file.write("\n")

            # Prepare DataFrame for statistics
            stats_columns = ['Event Level', 'Median Concentration', 'Average Concentration', 'Total Event Count', 'Total Event Days','Average Event Duration']
            stats_data = []
            for level_key, data in level_data.items():
                if data:
                    df_level_data = pd.DataFrame(data, columns=['new_corrected'])
                    median_concentration = df_level_data['new_corrected'].median()
                    average_concentration = df_level_data['new_corrected'].mean()
                else:
                    median_concentration = average_concentration = 0

                if event_stats[level_key]['count'] > 0 and event_stats[level_key]['count'] is not None:
                    ave_event_duration = event_stats[level_key]['total_days']/event_stats[level_key]['count']

                else: 
                    ave_event_duration = 0 
                ## This is turning data tyoe into string we need them as in

                stats_data.append([
                    level_key.capitalize(),
                    f"{median_concentration:.2f}",
                    f"{average_concentration:.2f}",
                    event_stats[level_key]['count'],
                    f"{event_stats[level_key]['total_days']:.2f}",
                    f"{ave_event_duration:.2f}",

                ])


            stats_df = pd.DataFrame(stats_data, columns=stats_columns)
            file.write(stats_df.to_csv(index=False))
            file.write("\n\n")
            general_event_data_per_location.append(stats_df)



    # Indicate completion and file location
    print(f"Data saved to {output_file_path}")
    return general_event_data_per_location

def data_for_annimated_bar_graph_df(new_dfs,name_label):

    output_dir = 'Data For Animated Bar Graphs'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, 'Concentration_animation_data.csv')


    prepared_dfs = []

    for df, label in zip(new_dfs, name_label):

        df_copy = df.copy()
        df_copy.set_index('datetime_utc', inplace=True)
        # Extract only the 'new_corrected' column and name it according to the sensor label
        prepared_df = df_copy[['new_corrected']].rename(columns={'new_corrected': label})
        prepared_dfs.append(prepared_df)

    # Concatenate all the prepared DataFrames along columns
    combined_df = pd.concat(prepared_dfs, axis=1)

    # Handle any missing data, if necessary
    combined_df.fillna(0, inplace=True)  # Replace NaN with 0, or use another method as required

    combined_df.to_csv(csv_path)

    return combined_df


# Function to create the new DataFrame with event flags
def create_event_flag_df(combined_df, event_detection_level_1, event_detection_level_2, event_detection_level_3):

    output_dir = 'Data For Animated Bar Graphs'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, 'Events_animation_data.csv')

    new_df = pd.DataFrame(0, index=combined_df.index, columns=combined_df.columns)

    for index, df_c in enumerate(event_detection_level_3):
        print(df_c)



        for _, row in df_c.iterrows():
            start_time = row['Start Time']
            end_time = row['End Time']
            location = row['Sensor Location']
        
            mask = (combined_df.index >= start_time) & (combined_df.index <= end_time)
            new_df.loc[mask, location] = 1

    for index, df_c in enumerate(event_detection_level_2):
        



        for _, row in df_c.iterrows():
            start_time = row['Start Time']
            end_time = row['End Time']
            location = row['Sensor Location']
        
            mask = (combined_df.index >= start_time) & (combined_df.index <= end_time)
            new_df.loc[mask, location] = 2

    for index, df_c in enumerate(event_detection_level_1):
        



        for _, row in df_c.iterrows():
            start_time = row['Start Time']
            end_time = row['End Time']
            location = row['Sensor Location']
        
            mask = (combined_df.index >= start_time) & (combined_df.index <= end_time)
            new_df.loc[mask, location] = 3

    new_df.to_csv(csv_path)

    return new_df



def event_bar_animation(df,Tabe_Title,Y_axis_Label):



    if not df.empty: 
        colors = plt.cm.viridis(np.linspace(0, 1, len(df.columns)))

        fig, ax = plt.subplots()
        bars = ax.bar(df.columns, df.iloc[0], color=colors)

        lines = [Line2D([], [], color='black', linewidth=0.5) for _ in range(len(bars)-1)]
        for line in lines:
            ax.add_line(line)

        def update(frame_number):
            y_data = df.iloc[frame_number]
            for bar, height in zip(bars, y_data):
                bar.set_height(height)

            # Update lines to connect the bars
            for i, line in enumerate(lines):

                if y_data[i+1] != 0 and len(y_data) > i+1 :

                    line.set_data([i, i+1], [y_data[i], y_data[i+1]])

                if y_data[i+1] == 0 and len(y_data) > i+2:
                    line.set_data([i, i+2], [y_data[i], y_data[i+2]])

                if y_data[i-1] != 0 and i > 0: 
                    line.set_data([i-1, i], [y_data[i-1], y_data[i]])

                if y_data[i-1] == 0 and i > 1: 
                    line.set_data([i-2, i], [y_data[i-2], y_data[i]])
            


            ax.set_ylim(0, df.values.max() + 1)  # Ensure the y-axis scales to the max value plus some space
            ax.set_title(f"Data for {df.index[frame_number].strftime('%Y-%m-%d %H:%M:%S')}")

        # Create the animation
        # Create the animation with reduced interval for faster playback
        ani = FuncAnimation(fig, update, frames=len(df), repeat=True, interval=10)  # 100 ms between frames

        # Enhance plot aesthetics
        ax.set_ylabel('New Corrected Value')
        ax.set_xlabel('Sensor Locations')
        plt.xticks(rotation=45, ha='right')  # Improve label visibility

        plt.show()

# Create the flagged DataFrame







'''

new_df_2 = pd.DataFrame(
    np.zeros_like(combined_df), 
    index=combined_df.index, 
    columns=combined_df.columns
)

'''

##Base Organized Data




'''

print(H1_H3_12H_Events_Dfs["level_3"][])



for data in H1_H3_12H_Events_Dfs:

    print(data["level_3"])
    print(data["level_2"])
    print(data[2])
quit()

'''




H12_Rolling_Data = rolling_window_event_data(new_dfs, .5)
H24_Rolling_Data = rolling_window_event_data(new_dfs, 1)
H36_Rolling_Data = rolling_window_event_data(new_dfs, 1.5)
H48_Rolling_Data = rolling_window_event_data(new_dfs, 2)




##different Event detectiom Parameters. 

#Time Window 

H1_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H12_Rolling_Data, .5, 3,1)
H1_H3_24H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H24_Rolling_Data, 1, 3,1)
H1_H3_36H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H36_Rolling_Data, 1.5, 3,1)
H1_H3_48H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H48_Rolling_Data, 2, 3,1)



#threshold duration to be considerd and event
H1_H1_5_24H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H24_Rolling_Data, 1, 1.5,1)
#H1_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H12_Rolling_Data, .5, 3,1)
H1_H6_24H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H24_Rolling_Data, 1, 6,1)
H1_H12_24H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H24_Rolling_Data, 1, 12,1)


#min Duration Between Events 

#H1_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs, H12_Rolling_Data, .5, 3,1)
H3_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H12_Rolling_Data, .5, 3,3)
H6_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H12_Rolling_Data, .5, 3,6)
H12_H3_12H_Events_Dfs = Processing_rolling_window_dfs(new_dfs,H12_Rolling_Data, .5, 3,12)




print(H1_H12_24H_Events_Dfs)



#Processing_rolling_window_dfs(original_dfs,dfs,event_length_in_days, hours_over_threshold_to_be_considered_an_event, hours_between_events)


Event_detection_parameters_list = [[.5,3,1],[1, 3,1],[1.5, 3,1],[2, 3,1],[1, 1.5,1],[1, 3,1],[1, 6,1],[1, 12,1],[.5, 3,1],[.5, 3,3],[.5, 3,6],[.5, 3,12]]
#event_detection_parameter_data = [H1_H3_12H_Events_Dfs,H1_H3_24H_Events_Dfs,]




event_detection_parameter_data = [
    H1_H3_12H_Events_Dfs,
    H1_H3_24H_Events_Dfs,
    H1_H3_36H_Events_Dfs,
    H1_H3_48H_Events_Dfs,
    

    H1_H1_5_24H_Events_Dfs,
    H1_H3_24H_Events_Dfs,
    H1_H6_24H_Events_Dfs,
    H1_H12_24H_Events_Dfs,

    H1_H3_12H_Events_Dfs,
    H3_H3_12H_Events_Dfs,
    H6_H3_12H_Events_Dfs,
    H12_H3_12H_Events_Dfs
    ]




#combined_df = data_for_annimated_bar_graph_df(new_dfs,name_label)

#flagged_df = create_event_flag_df(combined_df, H1_H3_12H_Events_Dfs['level_1'], H1_H3_12H_Events_Dfs['level_2'], H1_H3_12H_Events_Dfs['level_3'])

#event_bar_animation(combined_df,"Title","Yaxis")

#event_bar_animation(flagged_df,"Title","Yaxis")




#num_dfs = len(event_detection_parameter_data[0]['level_3'])  # Number of DataFrames in each list


# Initialize list to hold organized DataFrames
organized_dfs = [[] for _ in range(9)]  # Initialize list to hold organized DataFrames

# Organize the DataFrames
for dfs_list in event_detection_parameter_data:

    level_1 = []
    level_2 = []
    level_3 = []

    for count, df in enumerate(dfs_list['level_3']):
        level_3.append(df)

    for count, df in enumerate(dfs_list['level_2']):
        level_2.append(df)

    for count, df in enumerate(dfs_list['level_1']):
        level_1.append(df)

    print(level_3)
    print(level_2)
    print(level_1)

    for indexer,item in enumerate(level_1):
        quick_list = [level_1[indexer],level_2[indexer], level_3[indexer]]

        organized_dfs[indexer].append(quick_list)



    '''
    for i, df in enumerate(dfs_list):
        all_three_levels_per_location = [df[]]
        organized_df[df[]]
        organized_dfs[i].append(df)
    '''
output_dir = "Hope and Pray"
# Display the organized DataFrames
for i, dfs_group in enumerate(organized_dfs):
    #print(f"Group {i + 1} DataFrames:")
    #for a,df in enumerate(dfs_group):
    #print(f"Location:{name_label[i]}\n")
    # print(f"Event minimum Length in Days: {Event_detection_parameters_list[a][0]} Minimum duration above threshold (in Hours): {Event_detection_parameters_list[a][1]} Minimum Hours Between Events{Event_detection_parameters_list[a][2]}")
    #print(df)
    All_One_location_Scatter_Event_Graph_AllLevels_PDF_rolling_window(dfs_group, new_dfs, name_label, output_dir, percise_results_dfs, i,Event_detection_parameters_list )
        



Event_Range_Comparison = []


#rolling_window_result_12h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,.5,3)

output_directory = "Range Comparison H1_H3_12H"

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_12H_Events_Dfs['level_3'], 
    H1_H3_12H_Events_Dfs['level_2'], 
    H1_H3_12H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[0]

)

info = save_event_info_as_csv_5(
    H1_H3_12H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)




Event_Range_Comparison.append(info)





output_directory = "Range Comparison H1_H3_24H H"

#rolling_window_result_24h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,3)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_24H_Events_Dfs['level_3'], 
    H1_H3_24H_Events_Dfs['level_2'], 
    H1_H3_24H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[1]
)
info = save_event_info_as_csv_5(
    H1_H3_24H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Range_Comparison.append(info)


output_directory = "Range Comparison H1_H3_36H"

#rolling_window_result_36h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1.5,3)


# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_36H_Events_Dfs['level_3'], 
    H1_H3_36H_Events_Dfs['level_2'], 
    H1_H3_36H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[2]
)
info = save_event_info_as_csv_5(
    H1_H3_36H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Range_Comparison.append(info)

output_directory = "Range Comparison H1_H3_48H"


#rolling_window_result_48h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,2,3)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_48H_Events_Dfs['level_3'], 
    H1_H3_48H_Events_Dfs['level_2'], 
    H1_H3_48H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[3]
)
info = save_event_info_as_csv_5(
    H1_H3_48H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Range_Comparison.append(info)



SAFE_Happy_Camp_Community_Center_event_detection_params_all_data = []
SAFE_Swillup_Creek_event_detection_params_all_data = []
SAFE_Sandy_Bar_Creek_event_detection_params_all_data = []
SAFE_Somes_Bar_event_detection_params_all_data = []
Orleans_KDNR_Outdoor_event_detection_params_all_data = []
Butler_Creek_event_detection_params_all_data = []
Forks_Of_Salmon_event_detection_params_all_data = []
CARB_Cecilville_event_detection_params_all_data = []
CARB_Sawyers_Bar_event_detection_params_all_data = []

all_locations_event_detection_params_data = [
    SAFE_Happy_Camp_Community_Center_event_detection_params_all_data,
    SAFE_Swillup_Creek_event_detection_params_all_data,
    SAFE_Sandy_Bar_Creek_event_detection_params_all_data,
    SAFE_Somes_Bar_event_detection_params_all_data,
    Orleans_KDNR_Outdoor_event_detection_params_all_data,
    Butler_Creek_event_detection_params_all_data,
    Forks_Of_Salmon_event_detection_params_all_data,
    CARB_Cecilville_event_detection_params_all_data,
    CARB_Sawyers_Bar_event_detection_params_all_data
]

for  data_for_one_event_param in Event_Range_Comparison:




    for index_1, location in enumerate(data_for_one_event_param):
        print("\n\n")
        print(name_label[index_1])
        print("\n\n")
        print(location)

        all_locations_event_detection_params_data[index_1].append(location)


output_dir = "BAR GRAPH PLOTS"

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Setup PDF file to save plots
output_filepath = os.path.join(output_dir, 'H1_H3_12H24H36H48HEvent_Detection_Pararm_Comparison.pdf')
with PdfPages(output_filepath) as pdf:

    for index_2, locations_iterated in enumerate(all_locations_event_detection_params_data):
        print(f"\n\n{name_label[index_2]}\n")
        # Placeholder DataFrame to aggregate count data

        agg_count_data = pd.DataFrame()
        agg_mean_data = pd.DataFrame()
        agg_median_data = pd.DataFrame()
        agg_duration_data = pd.DataFrame()


        index_3_names = {0: '12H', 1: '24H', 2: '36H', 3: '48H'}
        
        for index_3, datas in enumerate(locations_iterated): 
            if not isinstance(datas, pd.DataFrame):
                datas = pd.DataFrame(datas, columns=['Event Level', 'Median Concentration', 'Average Concentration', 'Total Event Count', 'Total Event Days', 'Average Event Duration'])
                print(f"Converted to DataFrame: {datas}\n")
            else:
                print(f"Already a DataFrame: {datas}\n")

            # Ensure 'Event Level' is the index
            datas.set_index('Event Level', inplace=True)

            datas['Average Concentration'] = pd.to_numeric(datas['Average Concentration'], errors='coerce')
            
            datas['Median Concentration'] = pd.to_numeric(datas['Median Concentration'], errors='coerce')

            datas['Average Event Duration'] = pd.to_numeric(datas['Average Event Duration'], errors='coerce')


            datas.fillna(0, inplace=True)  # Handle NaNs by replacing them with zero

            # Rename column for the count to reflect the index for easier plotting
            count_column_name = f'Count {index_3_names[index_3]}'
            mean_column_name = f'Mean {index_3_names[index_3]}'
            median_column_name = f'Median {index_3_names[index_3]}'
            duration_column_name = f'Duration {index_3_names[index_3]}'



            datas.rename(columns={'Total Event Count': count_column_name, 'Average Concentration': mean_column_name,'Median Concentration':median_column_name, 'Average Event Duration': duration_column_name}, inplace=True)

            # Aggregate count data
            if agg_count_data.empty:
                agg_count_data = datas[[count_column_name]]
            else:
                agg_count_data = agg_count_data.join(datas[[count_column_name]], how='outer')

             # Aggregate count data
            if agg_mean_data.empty:
                agg_mean_data = datas[[mean_column_name]]
            else:
                agg_mean_data = agg_mean_data.join(datas[[mean_column_name]], how='outer')

            if agg_median_data.empty:
                agg_median_data = datas[[median_column_name]]
            else:
                agg_median_data = agg_median_data.join(datas[[median_column_name]], how='outer')

            if agg_duration_data.empty:
                agg_duration_data = datas[[duration_column_name]]
            else:
                agg_duration_data = agg_duration_data.join(datas[[duration_column_name]], how='outer')


        # Fill NaN values with 0, which might occur if some parameters don't have all event levels
        agg_count_data.fillna(0, inplace=True)
        agg_mean_data.fillna(0, inplace=True)
        agg_median_data.fillna(0, inplace=True)
        agg_duration_data.fillna(0, inplace=True)



        # Create a figure to contain the subplots
        fig, axs = plt.subplots(2, 2, figsize=(24, 20))  # Adjust figure size for more space

        # Plot for Total Event Count
        agg_count_data.plot.bar(ax=axs[0, 0], title=f'Event Count  - {name_label[index_2]}')
        axs[0, 0].set_xlabel('Event Level')
        axs[0, 0].set_ylabel('Total Event Count')
        axs[0, 0].tick_params(axis='x', rotation=0)
        axs[0, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Mean Concentration
        agg_mean_data.plot.bar(ax=axs[0, 1], title=f'Mean Event Concentration - {name_label[index_2]}')
        axs[0, 1].set_xlabel('Event Level')
        axs[0, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[0, 1].tick_params(axis='x', rotation=0)
        axs[0, 1].legend(title='Parameter Index', loc='upper left')

        # Plot for Average Event Duration
        agg_duration_data.plot.bar(ax=axs[1, 0], title=f'Average Duration  - {name_label[index_2]}')
        axs[1, 0].set_xlabel('Event Level')
        axs[1, 0].set_ylabel('Average Event Duration in hours')
        axs[1, 0].tick_params(axis='x', rotation=0)
        axs[1, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Median Concentration
        agg_median_data.plot.bar(ax=axs[1, 1], title=f'Median Event Concentration - {name_label[index_2]}')
        axs[1, 1].set_xlabel('Event Level')
        axs[1, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[1, 1].tick_params(axis='x', rotation=0)
        axs[1, 1].legend(title='Parameter Index', loc='upper left')

       
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1, hspace=0.4, wspace=0.2)
        plt.tight_layout(pad=4.0)  # Increase padding around the subplots

        # Set the super title with adjusted y position to avoid overlap, significantly higher
        fig.suptitle(f'{name_label[index_2]} Event Detection 3 Hours in 12, 24, 36, 48  Hour Window', fontsize=20, y=0.99)


        # Display the entire figure with all subplots
        pdf.savefig()  # Save the current figure into the pdf
        plt.close()
        # Display the entire figure with all subplots

    print(f"Saved all graphs into {output_filepath}")







Event_Duration_Range_Comparison = []


output_directory = "Range Comparison H1_H1_5_24H Events"

#rolling_window_result_24h_90min_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,1.5)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H1_5_24H_Events_Dfs['level_3'], 
    H1_H1_5_24H_Events_Dfs['level_2'], 
    H1_H1_5_24H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[4]

)
info = save_event_info_as_csv_5(
    H1_H1_5_24H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)

output_directory = "Range Comparison H1_H3_24H Events"

#rolling_window_result_24h_3h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,3)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_24H_Events_Dfs['level_3'], 
    H1_H3_24H_Events_Dfs['level_2'], 
    H1_H3_24H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[5]
)
info = save_event_info_as_csv_5(
    H1_H3_24H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)



output_directory = "Range H1_H6_24H Events"

#rolling_window_result_24h_6h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,6)


info = save_event_info_as_csv_5(
    H1_H6_24H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H6_24H_Events_Dfs['level_3'], 
    H1_H6_24H_Events_Dfs['level_2'], 
    H1_H6_24H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[6]
)

Event_Duration_Range_Comparison.append(info)



output_directory = "Range Comparison H1_H12_24H"

#rolling_window_result_24h_12h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,12)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H12_24H_Events_Dfs['level_3'], 
    H1_H12_24H_Events_Dfs['level_2'], 
    H1_H12_24H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[7]
)

info = save_event_info_as_csv_5(
    H1_H12_24H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)


SAFE_Happy_Camp_Community_Center_event_detection_params_all_data = []
SAFE_Swillup_Creek_event_detection_params_all_data = []
SAFE_Sandy_Bar_Creek_event_detection_params_all_data = []
SAFE_Somes_Bar_event_detection_params_all_data = []
Orleans_KDNR_Outdoor_event_detection_params_all_data = []
Butler_Creek_event_detection_params_all_data = []
Forks_Of_Salmon_event_detection_params_all_data = []
CARB_Cecilville_event_detection_params_all_data = []
CARB_Sawyers_Bar_event_detection_params_all_data = []

all_locations_event_detection_params_data = [
    SAFE_Happy_Camp_Community_Center_event_detection_params_all_data,
    SAFE_Swillup_Creek_event_detection_params_all_data,
    SAFE_Sandy_Bar_Creek_event_detection_params_all_data,
    SAFE_Somes_Bar_event_detection_params_all_data,
    Orleans_KDNR_Outdoor_event_detection_params_all_data,
    Butler_Creek_event_detection_params_all_data,
    Forks_Of_Salmon_event_detection_params_all_data,
    CARB_Cecilville_event_detection_params_all_data,
    CARB_Sawyers_Bar_event_detection_params_all_data
]


for  data_for_one_event_param in Event_Duration_Range_Comparison:




    for index_1, location in enumerate(data_for_one_event_param):
        print("\n\n")
        print(name_label[index_1])
        print("\n\n")
        print(location)

        all_locations_event_detection_params_data[index_1].append(location)


output_dir = "BAR GRAPH PLOTS"

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Setup PDF file to save plots
output_filepath = os.path.join(output_dir, 'H1_H15_H3_H6_H12_24H Event_Detection_Hours_Threshold_Pararm_Comparison.pdf')
with PdfPages(output_filepath) as pdf:

    for index_2, locations_iterated in enumerate(all_locations_event_detection_params_data):
        print(f"\n\n{name_label[index_2]}\n")
        # Placeholder DataFrame to aggregate count data

        agg_count_data = pd.DataFrame()
        agg_mean_data = pd.DataFrame()
        agg_median_data = pd.DataFrame()
        agg_duration_data = pd.DataFrame()


        index_3_names = {0: '1.5H', 1: '3H', 2: '6H', 3: '12H'}
        
        for index_3, datas in enumerate(locations_iterated): 
            if not isinstance(datas, pd.DataFrame):
                datas = pd.DataFrame(datas, columns=['Event Level', 'Median Concentration', 'Average Concentration', 'Total Event Count', 'Total Event Days', 'Average Event Duration'])
                print(f"Converted to DataFrame: {datas}\n")
            else:
                print(f"Already a DataFrame: {datas}\n")

            # Ensure 'Event Level' is the index
            datas.set_index('Event Level', inplace=True)

            datas['Average Concentration'] = pd.to_numeric(datas['Average Concentration'], errors='coerce')
            
            datas['Median Concentration'] = pd.to_numeric(datas['Median Concentration'], errors='coerce')

            datas['Average Event Duration'] = pd.to_numeric(datas['Average Event Duration'], errors='coerce')


            datas.fillna(0, inplace=True)  # Handle NaNs by replacing them with zero

            # Rename column for the count to reflect the index for easier plotting
            count_column_name = f'Count {index_3_names[index_3]}'
            mean_column_name = f'Mean {index_3_names[index_3]}'
            median_column_name = f'Median {index_3_names[index_3]}'
            duration_column_name = f'Duration {index_3_names[index_3]}'



            datas.rename(columns={'Total Event Count': count_column_name, 'Average Concentration': mean_column_name,'Median Concentration':median_column_name, 'Average Event Duration': duration_column_name}, inplace=True)

            # Aggregate count data
            if agg_count_data.empty:
                agg_count_data = datas[[count_column_name]]
            else:
                agg_count_data = agg_count_data.join(datas[[count_column_name]], how='outer')

             # Aggregate count data
            if agg_mean_data.empty:
                agg_mean_data = datas[[mean_column_name]]
            else:
                agg_mean_data = agg_mean_data.join(datas[[mean_column_name]], how='outer')

            if agg_median_data.empty:
                agg_median_data = datas[[median_column_name]]
            else:
                agg_median_data = agg_median_data.join(datas[[median_column_name]], how='outer')

            if agg_duration_data.empty:
                agg_duration_data = datas[[duration_column_name]]
            else:
                agg_duration_data = agg_duration_data.join(datas[[duration_column_name]], how='outer')


        # Fill NaN values with 0, which might occur if some parameters don't have all event levels
        agg_count_data.fillna(0, inplace=True)
        agg_mean_data.fillna(0, inplace=True)
        agg_median_data.fillna(0, inplace=True)
        agg_duration_data.fillna(0, inplace=True)



        # Create a figure to contain the subplots
        fig, axs = plt.subplots(2, 2, figsize=(24, 20))  # Adjust figure size for more space

        # Plot for Total Event Count
        agg_count_data.plot.bar(ax=axs[0, 0], title=f'Event Count  - {name_label[index_2]}')
        axs[0, 0].set_xlabel('Event Level')
        axs[0, 0].set_ylabel('Total Event Count')
        axs[0, 0].tick_params(axis='x', rotation=0)
        axs[0, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Mean Concentration
        agg_mean_data.plot.bar(ax=axs[0, 1], title=f'Mean Event Concentration - {name_label[index_2]}')
        axs[0, 1].set_xlabel('Event Level')
        axs[0, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[0, 1].tick_params(axis='x', rotation=0)
        axs[0, 1].legend(title='Parameter Index', loc='upper left')

        # Plot for Average Event Duration
        agg_duration_data.plot.bar(ax=axs[1, 0], title=f'Average Duration  - {name_label[index_2]}')
        axs[1, 0].set_xlabel('Event Level')
        axs[1, 0].set_ylabel('Average Event Duration in hours')
        axs[1, 0].tick_params(axis='x', rotation=0)
        axs[1, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Median Concentration
        agg_median_data.plot.bar(ax=axs[1, 1], title=f'Median Event Concentration - {name_label[index_2]}')
        axs[1, 1].set_xlabel('Event Level')
        axs[1, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[1, 1].tick_params(axis='x', rotation=0)
        axs[1, 1].legend(title='Parameter Index', loc='upper left')

                
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1, hspace=0.4, wspace=0.2)
        plt.tight_layout(pad=4.0)  # Increase padding around the subplots

        # Set the super title with adjusted y position to avoid overlap, significantly higher
        fig.suptitle(f'{name_label[index_2]} Event Detection 1.5 , 3, 6, 12 Hours in 24 Hour Window', fontsize=20, y=0.99)



        # Display the entire figure with all subplots
        pdf.savefig()  # Save the current figure into the pdf
        plt.close()

    print(f"Saved all graphs into {output_filepath}")





Event_Duration_Range_Comparison = []

output_directory = "Range Comparison 1H 3H 24H Events"

#rolling_window_result_24h_3h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,3)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H1_H3_12H_Events_Dfs['level_3'], 
    H1_H3_12H_Events_Dfs['level_2'], 
    H1_H3_12H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[8]
)
info = save_event_info_as_csv_5(
    H3_H3_12H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)


output_directory = "Range Comparison 3H 3H 24H Events"

#rolling_window_result_24h_3h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,3)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H3_H3_12H_Events_Dfs['level_3'], 
    H3_H3_12H_Events_Dfs['level_2'], 
    H3_H3_12H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[9]
)
info = save_event_info_as_csv_5(
    H3_H3_12H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)



output_directory = "Range Comparison 6H 3H 24H Events"

#rolling_window_result_24h_6h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,6)


info = save_event_info_as_csv_5(
    H6_H3_12H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H6_H3_12H_Events_Dfs['level_3'], 
    H6_H3_12H_Events_Dfs['level_2'], 
    H6_H3_12H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[10]
)

Event_Duration_Range_Comparison.append(info)



output_directory = "Range Comparison 12H 3H 24H Events"

#rolling_window_result_24h_12h_event_detection = rolling_window_event_analysis_all_3_levels_with_median_and_avg(new_dfs,1,12)

# Call the function with the specified output directory
Scatter_Event_Graph_AllLevels_PDF_rolling_window(
    H12_H3_12H_Events_Dfs['level_3'], 
    H12_H3_12H_Events_Dfs['level_2'], 
    H12_H3_12H_Events_Dfs['level_1'], 
    new_dfs, 
    name_label, 
    output_directory, 
    percise_results_dfs,
    Event_detection_parameters_list[11]
)
info = save_event_info_as_csv_5(
    H12_H3_12H_Events_Dfs,
    new_dfs,  # The original dataframes used for concentration calculations
    name_label,
    output_directory
)

Event_Duration_Range_Comparison.append(info)


SAFE_Happy_Camp_Community_Center_event_detection_params_all_data = []
SAFE_Swillup_Creek_event_detection_params_all_data = []
SAFE_Sandy_Bar_Creek_event_detection_params_all_data = []
SAFE_Somes_Bar_event_detection_params_all_data = []
Orleans_KDNR_Outdoor_event_detection_params_all_data = []
Butler_Creek_event_detection_params_all_data = []
Forks_Of_Salmon_event_detection_params_all_data = []
CARB_Cecilville_event_detection_params_all_data = []
CARB_Sawyers_Bar_event_detection_params_all_data = []

all_locations_event_detection_params_data = [
    SAFE_Happy_Camp_Community_Center_event_detection_params_all_data,
    SAFE_Swillup_Creek_event_detection_params_all_data,
    SAFE_Sandy_Bar_Creek_event_detection_params_all_data,
    SAFE_Somes_Bar_event_detection_params_all_data,
    Orleans_KDNR_Outdoor_event_detection_params_all_data,
    Butler_Creek_event_detection_params_all_data,
    Forks_Of_Salmon_event_detection_params_all_data,
    CARB_Cecilville_event_detection_params_all_data,
    CARB_Sawyers_Bar_event_detection_params_all_data
]


for  data_for_one_event_param in Event_Duration_Range_Comparison:




    for index_1, location in enumerate(data_for_one_event_param):
        print("\n\n")
        print(name_label[index_1])
        print("\n\n")
        print(location)

        all_locations_event_detection_params_data[index_1].append(location)


output_dir = "BAR GRAPH PLOTS"

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Setup PDF file to save plots
output_filepath = os.path.join(output_dir, 'Event_Detection_minimumduration_between_events.pdf')
with PdfPages(output_filepath) as pdf:

    for index_2, locations_iterated in enumerate(all_locations_event_detection_params_data):
        print(f"\n\n{name_label[index_2]}\n")
        # Placeholder DataFrame to aggregate count data

        agg_count_data = pd.DataFrame()
        agg_mean_data = pd.DataFrame()
        agg_median_data = pd.DataFrame()
        agg_duration_data = pd.DataFrame()


        index_3_names = {0: '1H ', 1: '3H', 2: '6H', 3: '12H'}
        
        for index_3, datas in enumerate(locations_iterated): 
            if not isinstance(datas, pd.DataFrame):
                datas = pd.DataFrame(datas, columns=['Event Level', 'Median Concentration', 'Average Concentration', 'Total Event Count', 'Total Event Days', 'Average Event Duration'])
                print(f"Converted to DataFrame: {datas}\n")
            else:
                print(f"Already a DataFrame: {datas}\n")

            # Ensure 'Event Level' is the index
            datas.set_index('Event Level', inplace=True)

            datas['Average Concentration'] = pd.to_numeric(datas['Average Concentration'], errors='coerce')
            
            datas['Median Concentration'] = pd.to_numeric(datas['Median Concentration'], errors='coerce')

            datas['Average Event Duration'] = pd.to_numeric(datas['Average Event Duration'], errors='coerce')


            datas.fillna(0, inplace=True)  # Handle NaNs by replacing them with zero

            # Rename column for the count to reflect the index for easier plotting
            count_column_name = f'Count {index_3_names[index_3]}'
            mean_column_name = f'Mean {index_3_names[index_3]}'
            median_column_name = f'Median {index_3_names[index_3]}'
            duration_column_name = f'Duration {index_3_names[index_3]}'



            datas.rename(columns={'Total Event Count': count_column_name, 'Average Concentration': mean_column_name,'Median Concentration':median_column_name, 'Average Event Duration': duration_column_name}, inplace=True)

            # Aggregate count data
            if agg_count_data.empty:
                agg_count_data = datas[[count_column_name]]
            else:
                agg_count_data = agg_count_data.join(datas[[count_column_name]], how='outer')

             # Aggregate count data
            if agg_mean_data.empty:
                agg_mean_data = datas[[mean_column_name]]
            else:
                agg_mean_data = agg_mean_data.join(datas[[mean_column_name]], how='outer')

            if agg_median_data.empty:
                agg_median_data = datas[[median_column_name]]
            else:
                agg_median_data = agg_median_data.join(datas[[median_column_name]], how='outer')

            if agg_duration_data.empty:
                agg_duration_data = datas[[duration_column_name]]
            else:
                agg_duration_data = agg_duration_data.join(datas[[duration_column_name]], how='outer')


        # Fill NaN values with 0, which might occur if some parameters don't have all event levels
        agg_count_data.fillna(0, inplace=True)
        agg_mean_data.fillna(0, inplace=True)
        agg_median_data.fillna(0, inplace=True)
        agg_duration_data.fillna(0, inplace=True)



        # Create a figure to contain the subplots
        fig, axs = plt.subplots(2, 2, figsize=(24, 20))  # Adjust figure size for more space

        # Plot for Total Event Count
        agg_count_data.plot.bar(ax=axs[0, 0], title=f'Event Count  - {name_label[index_2]}')
        axs[0, 0].set_xlabel('Event Level')
        axs[0, 0].set_ylabel('Total Event Count')
        axs[0, 0].tick_params(axis='x', rotation=0)
        axs[0, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Mean Concentration
        agg_mean_data.plot.bar(ax=axs[0, 1], title=f'Mean Event Concentration - {name_label[index_2]}')
        axs[0, 1].set_xlabel('Event Level')
        axs[0, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[0, 1].tick_params(axis='x', rotation=0)
        axs[0, 1].legend(title='Parameter Index', loc='upper left')

        # Plot for Average Event Duration
        agg_duration_data.plot.bar(ax=axs[1, 0], title=f'Average Duration  - {name_label[index_2]}')
        axs[1, 0].set_xlabel('Event Level')
        axs[1, 0].set_ylabel('Average Event Duration in hours')
        axs[1, 0].tick_params(axis='x', rotation=0)
        axs[1, 0].legend(title='Parameter Index', loc='upper left')

        # Plot for Median Concentration
        agg_median_data.plot.bar(ax=axs[1, 1], title=f'Median Event Concentration - {name_label[index_2]}')
        axs[1, 1].set_xlabel('Event Level')
        axs[1, 1].set_ylabel('Particulate Matter < 2.5 Microns Concentration (µg/m³)')
        axs[1, 1].tick_params(axis='x', rotation=0)
        axs[1, 1].legend(title='Parameter Index', loc='upper left')

                
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1, hspace=0.4, wspace=0.2)
        plt.tight_layout(pad=4.0)  # Increase padding around the subplots

        # Set the super title with adjusted y position to avoid overlap, significantly higher
        fig.suptitle(f'{name_label[index_2]} Event Detection 1.5 , 3, 6, 12 Hours in 24 Hour Window', fontsize=20, y=0.99)



        # Display the entire figure with all subplots
        pdf.savefig()  # Save the current figure into the pdf
        plt.close()

    print(f"Saved all graphs into {output_filepath}")





quit()



