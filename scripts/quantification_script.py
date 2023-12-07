import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def detect_delimiter(csv_file_path):
    with open(csv_file_path, 'r') as f:
        first_line = f.readline()
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(first_line).delimiter
    return delimiter


def main():
    parser = argparse.ArgumentParser(description="Quantification Script")
    parser.add_argument("--data_directory", required=True,
                        help="Directory containing all .csv files to be processed.")
    parser.add_argument("--standards_csv", required=True,
                        help="Path to standards.csv file.")
    parser.add_argument("--setup_csv", required=True,
                        help="Path to setup.csv file.")
    parser.add_argument("--save_histograms", action="store_true",
                        help="Save histograms for each channel.")
    parser.add_argument("--save_channel_data", action="store_true",
                        help="Save quantified data for each channel as a separate .csv file.")
    parser.add_argument("--save_combined_data", action="store_true",
                        help="Save all quantified data into a single .csv file.")

    args = parser.parse_args()

    channels_delimiter = detect_delimiter(args.setup_csv)
    channels_df = pd.read_csv(args.setup_csv, delimiter=channels_delimiter)

    standards_delimiter = detect_delimiter(args.standards_csv)
    standards_df = pd.read_csv(
        args.standards_csv, delimiter=standards_delimiter)

    merged_config_df = pd.merge(
        channels_df, standards_df, on='standard', how='left')

    output_dir = os.path.join(args.data_directory, 'output')
    graphs_dir = os.path.join(output_dir, 'graphs')
    csvs_dir = os.path.join(output_dir, 'CSVs')

    # Create output directories if they don't exist
    for directory in [output_dir, graphs_dir, csvs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for data_filename in os.listdir(args.data_directory):
        if not data_filename.endswith('.csv'):
            continue

        data_filepath = os.path.join(args.data_directory, data_filename)
        data_delimiter = detect_delimiter(data_filepath)
        data_df = pd.read_csv(data_filepath, delimiter=data_delimiter)

        for index, row in merged_config_df.iterrows():
            channel = row['channel']
            slope = row['slope']
            should_quantify = row['quantify']
            should_histogram = row['histogram']

            new_column_name = channel

            if should_quantify and not pd.isna(slope):
                data_df[channel] = data_df[channel] / slope
                new_column_name = f"{channel}_quantified"
                data_df.rename(
                    columns={channel: new_column_name}, inplace=True)

            if should_histogram and args.save_histograms:
                sns.displot(data_df[new_column_name], kde=False, color='blue')
                plt.title(f'Histogram of {new_column_name}')
                plt.xlabel(new_column_name)
                plt.ylabel('Frequency')
                sns.despine()
                plt.savefig(os.path.join(graphs_dir, f"{new_column_name}_histogram.png"), dpi=600)
                plt.close()

        if args.save_combined_data:
            combined_data_save_path = os.path.join(csvs_dir, f"{data_filename.split('.')[0]}_combined_quantified.csv")
            data_df.to_csv(combined_data_save_path, index=False)


if __name__ == "__main__":
    main()
