# pdf_generation.py
import os                   # For checking and creating directories
import pandas as pd         # For handling DataFrames and CSV operations
import numpy as np          # For numerical operations (e.g., np.select)
import matplotlib.pyplot as plt   # For plotting
from matplotlib.backends.backend_pdf import PdfPages  # For saving plots as PDF
from matplotlib.lines import Line2D                   # For custom legends
import matplotlib.patches as mpatches                 # For custom legend patches
from datetime import timedelta  # For handling time intervals


from matplotlib import font_manager



import csv                      


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

def Scatter_Event_Graph_AllLevels_PDF_rolling_window(level_5_events_detected,level_4_events_detected,level_3_events_detected, level_2_events_detected, level_1_events_detected, new_dfs, name_label, output_dir, params, nick_names,is_2022=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    days = int(params[0])
    duration = params[1]
    min_time_between_events = params[2]


    levels_detected = [level_1_events_detected, level_2_events_detected, level_3_events_detected,level_4_events_detected,level_5_events_detected]
    level_keys = [1, 2, 3,4,5]
    level_colors = ['yellow', 'orange', 'red', 'purple', 'maroon']

    overall_min_date = min(df['datetime_utc'].min() for df in new_dfs)
    overall_max_date = max(df['datetime_utc'].max() for df in new_dfs)

    # Setup PDF file to save plots
    File_Name = f'Window{days}_Threshold{duration}_LULL{min_time_between_events}_locations_events.pdf'
    output_filepath = os.path.join(output_dir, File_Name)

    with PdfPages(output_filepath) as pdf:
        for i, df in enumerate(new_dfs):
            plt.figure(figsize=(14, 6))
            conditions = [
                (df['new_corrected'] >= 225.5),
                (df['new_corrected'] >= 125.5) & (df['new_corrected'] <= 225.4),
                (df['new_corrected'] >= 55.5) & (df['new_corrected'] <= 125.4),
                (df['new_corrected'] >= 35.5) & (df['new_corrected'] <= 55.4),
                (df['new_corrected'] >= 9.1) & (df['new_corrected'] <= 35.4),
                (df['new_corrected'] >= 0.0) & (df['new_corrected'] <= 9.1)
            ]
            colors = ['maroon', 'purple', 'red', 'orange', 'yellow', 'green']
            #colors = ['black', 'black', 'black', 'black', 'black', 'black']
            df.loc[:, 'color'] = np.select(conditions, colors, default='blue')

            #plt.plot(percise_results_dfs[i]['datetime_utc'], percise_results_dfs[i]['Average PM2.5'], 2, color="black", alpha=1)

            for color in colors:
                subset = df[df['color'] == color]
                if is_2022:
                    plt.scatter(subset['datetime_utc'], subset['new_corrected'], 0.8, color=color, alpha=.5)

                else: 
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
                        event_bar_heights = [min(35.5, float(highest_pm25)),min(55.5, float(highest_pm25)),min(125.5, float(highest_pm25)),min(225.5, float(highest_pm25)),min(799, float(highest_pm25))]
                        event_bar_bottoms = [0,35.5,55.5, 125.5,225.5]
                        bar_widths = [1, 1, 1,1,1]

                        if is_2022:
                            plt.plot([start_time, start_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx]*1.5, color='gray', alpha=.3)
                            plt.plot([start_time, end_time], [event_bar_heights[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx]*1.5, color='gray', alpha=.3)
                            plt.plot([end_time, end_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx]*1.5, color='gray', alpha=.3)


                        plt.plot([start_time, start_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=1)
                        plt.plot([start_time, end_time], [event_bar_heights[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=1)
                        plt.plot([end_time, end_time], [event_bar_bottoms[level_idx], event_bar_heights[level_idx]], linewidth=bar_widths[level_idx], color=level_colors[level_idx], alpha=1)


            #plt.title(f'PM$_{{2.5}}$ Concentrations and Events for {name_label[i]} 2021')
            #plt.title(f'Observation Window:{days} H   Threshold Exceedance Time:{duration} H  Lull Time: {min_time_between_events} H')
            #plt.suptitle(f'PM$_{{2.5}}$ Concentrations and Events for {nick_names[name_label[i]]} 2021')
            plt.title(f'{nick_names[name_label[i]]} (2021): PM$_{{2.5}}$ Concentrations\n'
            f'Window: {days} H | Threshold: {duration} H | Lull: {min_time_between_events} H', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('PM$_{2.5}$ Concentration (µg/m³)', fontsize=12)
            plt.xlim(overall_min_date, overall_max_date)

            if not is_2022:
                plt.ylim(0, 800)
            else:
                plt.ylim(0, 300)

            legend_handles = [
                mpatches.Patch(color='maroon', label='Class 5 (>225.5)'),
                mpatches.Patch(color='purple', label='Class 4 (>125.5)'),
                mpatches.Patch(color='red', label='Class 3 (>55.5)'),
                mpatches.Patch(color='orange', label='Class 2 (>35.5)'),
                mpatches.Patch(color='yellow', label='Class 1 (>9.1)')
                #mpatches.Patch(color='green', label='(0 - 12.0)')
            ]

            # First legend in the upper left
            first_legend = plt.legend(handles=legend_handles, loc='upper right', title='Event Bars ', frameon=True)

            # Second legend (dots)
            legend_handles_2 = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='maroon', markersize=10, label='>225.5'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='125.5 - 225.4'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='55.5 - 125.4'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='35.5 - 55.4'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='9.1 - 35.4'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='0 - 9.1.0')
            ]

            # Second legend slightly below the first legend
            second_legend = plt.legend(handles=legend_handles_2, loc='upper right', bbox_to_anchor=(1, 0.72), title='Air Quality Data Points', frameon=True)

            # Add the first legend back manually to avoid being overwritten
            plt.gca().add_artist(first_legend)



            plt.tight_layout(pad=2.0)

            file_name = f'{nick_names[name_label[i]]}_Window{days}H_Threshold{duration}H_LULL{min_time_between_events}H.png'
            output_path = os.path.join(output_dir, file_name)
            plt.savefig(output_path, dpi=300)


            pdf.savefig()  # Save the current figure into the pdf
            plt.close()

    print(f"Saved all graphs into {output_filepath}")




def save_event_info_as_csv_5(consolidated_results, original_dfs, names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the levels and their numeric equivalents
    levels = {'level_5': 5,'level_4': 4,'level_3': 3, 'level_2': 2, 'level_1': 1}

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
                df_cleaned['Event Duration (days)'] = (df_cleaned['End Time'] - df_cleaned['Start Time']).dt.total_seconds() / 86400

                combined_df = pd.concat([combined_df, df_cleaned], ignore_index=True)

                # Collect data and calculate stats for each event
                for _, event in df_cleaned.iterrows():

                    mask = (original_dfs[i]['datetime_utc'] >= event['Start Time']) & (original_dfs[i]['datetime_utc'] <= event['End Time'])

                    level_data[level_key].extend(original_dfs[i].loc[mask, 'new_corrected'].tolist())


                    event_stats[level_key]['count'] += 1
                    event_stats[level_key]['total_days'] += (event['End Time'] - event['Start Time']).total_seconds() / 86400  # Convert seconds to days

            # Sort by Start Time and Event Level

            print(combined_df.columns)

            if combined_df.empty:  # Correct method to check if DataFrame is empty
                print("combined_df is empty. Skipping sorting.")
                continue
            else:
                # Ensure Start Time is in datetime format
                combined_df['Start Time'] = pd.to_datetime(combined_df['Start Time'], errors='coerce')

                # Drop rows with missing Start Time
                combined_df = combined_df.dropna(subset=['Start Time'])

                # Sort by Start Time and Event Level
                sorted_combined = combined_df.sort_values(by=['Start Time', 'Event Level'])
                print("Sorted DataFrame:\n", sorted_combined.head())


            
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


def process_event_data( Event_Range_Comparison):
    """
    Processes and organizes event data by locations.

    Parameters:
        sensor_dfs (list): List of DataFrames representing sensor data.
        Event_Range_Comparison (list): Event data comparisons for various locations.

    Returns:
        all_locations_event_detection_params_data
    """
    #print("\n### Step 1: Extracting parameter values ###")
    #param_values_list = [list(params.values()) for params in Event_detection_parameters_list]
    #print(f"Extracted param_values_list: {param_values_list}")

    print("\n### Step 2: Initializing location-based event data structure ###")
    num_locations = len(Event_Range_Comparison[0])  # Assuming all sublists have the same number of locations
    all_locations_event_detection_params_data = [[] for _ in range(num_locations)]
    print(f"Initialized empty all_locations_event_detection_params_data with {num_locations} locations.")

    print("\n### Step 3: Processing and organizing event data by location ###")
    for data_for_one_event_param in Event_Range_Comparison:
        print(f"Processing data_for_one_event_param: {data_for_one_event_param}")
        for index_1, location in enumerate(data_for_one_event_param):
            print(f"Appending location data to index {index_1}: {location}")
            all_locations_event_detection_params_data[index_1].append(location)

    print(f"Final all_locations_event_detection_params_data: {all_locations_event_detection_params_data}")

    return all_locations_event_detection_params_data


def generate_event_graphs_dynamic(
    location_event_data,       # Event data organized by location
    location_labels,           # Labels for each location
    nick_names,
    output_directory,          # Directory for saving files
    varying_suffixes,          # Mapping of indices to suffix names for the varying parameter
    legend_labels,             # Custom legend labels
    varying_param_name,
    constant_parameters      # Name of the varying parameter (e.g., 'Threshold', 'Lull Time', 'Observation Window')
    ):
    """
    Generates bar graphs for event data and saves them as PNG files and a consolidated PDF.
    Works dynamically for any varying parameter (e.g., thresholds, lull times, or observation windows).

    Parameters:
        location_event_data (list): Event data organized by location.
        location_labels (list): Labels for each location.
        output_directory (str): Directory to save the output files.
        varying_suffixes (dict): Mapping of indices to suffix names for the varying parameter.
        legend_labels (list): Custom labels for the legend in the plots.
        varying_param_name (str): Name of the varying parameter to display in legends and titles.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Define the consolidated PDF file path
    pdf_file_path = os.path.join(output_directory, f'Event_Graphs_By_{varying_param_name.replace(" ", "_")}.pdf')

    with PdfPages(pdf_file_path) as pdf_writer:
        for location_index, location_data in enumerate(location_event_data):
            print(f"\n### Processing data for location: {location_labels[location_index]} ###")

            # Initialize empty DataFrames for aggregating metrics
            aggregated_metrics = {
                'count': pd.DataFrame(),
                'mean': pd.DataFrame(),
                'duration': pd.DataFrame(),
                'median': pd.DataFrame()
            }
            print("Initialized empty aggregated metrics:", aggregated_metrics.keys())

            # Process each varying parameter's data
            for index, event_data in enumerate(location_data):
                param_label = varying_suffixes[index]
                print(f"\n--- Processing {varying_param_name} {param_label} ---")

                print(event_data.head())

                event_df = pd.DataFrame(event_data, columns=[
                    'Event Level',
                    'Median Concentration',
                    'Average Concentration',
                    'Total Event Count',
                    'Total Event Days',
                    'Average Event Duration'
                ])
                event_df.set_index('Event Level', inplace=True)
                event_df = event_df.apply(pd.to_numeric, errors='coerce').fillna(0)

                event_df.rename(columns={
                    'Total Event Count': f'Count {param_label}',
                    'Average Concentration': f'Mean {param_label}',
                    'Average Event Duration': f'Duration {param_label}',
                    'Median Concentration': f'Median {param_label}'
                }, inplace=True)

                for metric, agg_df in aggregated_metrics.items():
                    column_name = f"{metric.capitalize()} {param_label}"
                    if agg_df.empty:
                        aggregated_metrics[metric] = event_df[[column_name]]
                        print(f"Initialized {metric} DataFrame with column {column_name} for {varying_param_name} {param_label}:\n", aggregated_metrics[metric].head())
                    else:
                        aggregated_metrics[metric] = agg_df.join(event_df[[column_name]], how='outer')
                        print(f"Updated {metric} DataFrame with column {column_name} for {varying_param_name} {param_label}:\n", aggregated_metrics[metric].head())

                for metric_name, metric_df in aggregated_metrics.items():
                    metric_df.fillna(0, inplace=True)
                    print(f"Final {metric_name} DataFrame after filling missing values:\n", metric_df.head())

            # Create subplots for the metrics
            fig, axs = plt.subplots(2, 2, figsize=(24, 24))
            fig.patch.set_facecolor('#FFFFFF')  # Set the background color

            titles = [
                'Event Count',
                'Mean Event Concentration',
                'Average Event Duration',
                'Median Event Concentration'
            ]
            y_labels = [
                'Total Event Count',
                'µg/m³',
                'Average Event Duration in Days',
                'µg/m³'
            ]

            for i, (metric_name, ax) in enumerate(zip(aggregated_metrics.keys(), axs.flatten())):
                aggregated_metrics[metric_name].plot.bar(ax=ax)
                ax.set_title(f"{titles[i]} - {nick_names[location_labels[location_index]]}", fontsize=20, weight='bold')

                # Customize axis labels and ticks
                custom_labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
                ax.set_xticks(range(len(custom_labels)))
                ax.set_xticklabels(custom_labels, fontsize=16, rotation=0, ha='center')
                ax.set_xlabel('Event Level', fontsize=20, weight='semibold')
                ax.set_ylabel(y_labels[i], fontsize=20, weight='semibold')
                ax.tick_params(axis='both', which='major', labelsize=18)

                # Set background colors alternately
                if i in [0, 3]:
                    ax.set_facecolor('#f5f5f5')  # Light gray
                else:
                    ax.set_facecolor('#f0f8ff')  # Very light blue

                # Add legend
                ax.legend(
                    labels=legend_labels,
                    title=f'{varying_param_name}',
                    loc='upper center',
                    bbox_to_anchor=(0.5, 0.995),
                    frameon=True,
                    ncol=len(legend_labels),
                    prop=font_manager.FontProperties(weight='bold', size=16),
                    title_fontproperties=font_manager.FontProperties(weight='semibold', size=16)
                )

                # Adjust y-axis to include a buffer
                max_val = aggregated_metrics[metric_name].max().max()
                ax.set_ylim(top=max_val * 1.2)

            # Finalize figure layout
            plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.08, hspace=0.3, wspace=0.3)
            fig.suptitle(f"{nick_names[location_labels[location_index]]} Comparative Bar Graphs For {varying_param_name} (2021)", fontsize=36, weight='bold', y=0.975)
            fig.text(
                0.5, 0.93,
                f"|| Variable: {varying_param_name} ({', '.join(legend_labels)})|| Constants: {', '.join([f'{key}: {value}' for key, value in constant_parameters.items()])} ||",
                fontsize=26, weight='semibold', ha='center', va='center', color='dimgray'
                )

            # Save as PNG and add to PDF
            png_file_name = f"{nick_names[location_labels[location_index]]}_Event_Graphs_By_{varying_param_name.replace(' ', '_')}.png"
            png_file_path = os.path.join(output_directory, png_file_name)
            plt.savefig(png_file_path, dpi=300)
            pdf_writer.savefig(fig)
            plt.close()

        print(f"\nSaved all graphs into {pdf_file_path}")


