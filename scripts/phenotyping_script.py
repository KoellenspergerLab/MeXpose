# Importing relevant modules
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import phenograph
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from tifffile import imread
import umap


def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
    - argparse.Namespace object containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Process all CSV files within a directory.')

    parser.add_argument('-f', '--folder', type=str, required=True,
                        help='The directory containing the CSV and _mask files.')
    parser.add_argument('-wd', '--working_directory', type=str, default=None,
                        help='Directory where processed data and/or graphs will be saved. Defaults to the given folder if not provided.')
    parser.add_argument('--setup_csv', type=str, default=None,
                        help='Path to a CSV containing channels to normalize. First column should have channel names, second should have 1 for "yes" or 0 for "no".')
    parser.add_argument('--save_histograms', action='store_true', default=False,
                        help='Flag to save histograms of all specified columns for each CSV. If unset, defaults to "False".')
    parser.add_argument('--save_processed_histograms', action='store_true', default=False,
                        help='Flag to save histograms of processed columns for each CSV. If unset, defaults to "False".')
    parser.add_argument('--pixel_size', type=float, default=None,
                        help='Pixel size in µm for normalization.')
    parser.add_argument('--outlier_filtering_method', type=str,
                        choices=['percentiles', 'zscore'], default=None,
                        help='Method for outlier filtering. Can be "percentiles" or "zscore". Defaults to "percentiles".')
    parser.add_argument('--percentiles', type=str, default="0.0,0.997",
                        help='Comma-separated percentile values for percentile-based outlier removal. Format is "p_low,p_high". Defaults to "0.0,0.997".')
    parser.add_argument('--n_std', type=float, default=3,
                        help='Number of standard deviations for z-score-based outlier removal. Defaults to 3.')
    parser.add_argument('--save_processed_csv', action='store_true', default=False,
                        help='Flag to save processed DataFrames as CSV files. If unset, defaults to "False".')
    parser.add_argument('--no_cluster', action='store_true', default=False,
                        help='Flag to disable clustering and scaling. If unset, defaults to "False".')
    parser.add_argument('--clustering_parameters', type=str, default="30,1.0,42",
                        help='Comma-separated list of clustering parameters in the form "k,resolution_parameter,seed". Defaults to "30,1.0,42".')
    parser.add_argument('--scaling_method', type=str, choices=['robust', 'minmax'], default='robust',
                        help='Choose the scaling method for clustering. Options are "robust" and "minmax". Default is "robust".')
    parser.add_argument('--no_umap', action='store_true', default=False,
                        help='Flag to disable UMAP dimensionality reduction. If unset, defaults to "False".')
    parser.add_argument('--umap_parameters', default='15,0.1,42', type=str,
                        help='UMAP parameters in the format: "n_neighbors,min_dist,random_state". Defaults to "15,0.1,42".')
    parser.add_argument("--aggregation_method", type=str, choices=["median", "mean"], default="median",
                        help="Method to aggregate data for each cluster, options are 'median' or 'mean'. Default is 'median'.")
    parser.add_argument('--save_individual_clusters', action='store_true', default=False,
                        help='Save individual cluster dataframes as separate CSV files. If unset, defaults to "False".')
    parser.add_argument('--save_combined_clusters', action='store_true', default=False,
                        help='Save the processed data with cluster information as a CSV file. If unset, defaults to "False".')

    return parser.parse_args()


def read_setup_csv(setup_csv_path, area_column):
    """
    Reads a secondary CSV file to determine which channels to use for various data processing steps.

    Parameters:
    - setup_csv_path: str, path to the secondary CSV file
    - area_column: str, default area column name to use if not specified in the CSV

    Returns:
    - channels_to_normalize: list of str, channels to normalize
    - channels_to_filter: list of str, channels to apply outlier filtering
    - area_column: str, name of the area column
    - channels_to_histogram: list of str, channels to generate histograms
    - channels_to_heatmap: list of str, channels to generate heatmaps
    - channels_to_cluster: list of str, channels to use for clustering
    - channels_to_cluster_heatmap = list of str, channels to generate cluster heatmaps

    Warnings:
    - Missing expected columns will trigger a warning and the respective steps will be skipped.
    - If the 'isArea' column contains more than one entry, a warning will be issued and the default area_column will be used.
    """
    try:
        channels_df = pd.read_csv(setup_csv_path, delimiter=',')
        expected_columns = ['channels', 'normalize', 'isArea',
                            'filter', 'histogram', 'heatmap', 'cluster', 'cluster heatmap']

        # Initialize empty lists or default values for each expected column
        channel_data = {
            'normalize': [],
            'filter': [],
            'histogram': [],
            'heatmap': [],
            'cluster': [],
            'cluster heatmap': [],
        }

        # Identify and warn about missing columns
        missing_columns = [
            col for col in expected_columns if col not in channels_df.columns]
        if missing_columns:
            print(f"WARNING: The following expected columns are missing: {missing_columns}.")
            print("The steps related to these missing columns will be skipped.")

        # Populate the channel_data dictionary for available columns
        for col in channel_data.keys():
            if col in channels_df.columns:
                channel_data[col] = channels_df[channels_df[col]
                                                == 1]['channels'].tolist()

        # Handle the 'isArea' column separately
        if 'isArea' in channels_df.columns:
            area_column_candidates = channels_df[channels_df['isArea'] == 1]['channels'].tolist(
            )
            if len(area_column_candidates) != 1:
                print(f"WARNING: There should be exactly one area column specified. Using default '{area_column}'.")
            else:
                area_column = area_column_candidates[0]

        return channel_data['normalize'], channel_data['filter'], area_column, channel_data['histogram'], channel_data['heatmap'], channel_data['cluster'], channel_data['cluster heatmap']

    except Exception as e:
        print(f"WARNING: Failed to read setup_csv due to {e}. All steps related to it will be skipped.")
        return [], [], area_column, [], [], [], []


def read_segmentation_mask(folder_path, csv_file):
    try:
        # List all files in the folder
        files_in_folder = os.listdir(folder_path)
    except FileNotFoundError:
        return None

    mask_file = next((f for f in files_in_folder if f.startswith(f"{csv_file.split('.')[0]}") and f.endswith('_mask.png')), None)

    if mask_file:
        mask_path = os.path.join(folder_path, mask_file)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is not None:
            return mask_img

    return None


def read_overlay_image(overlay_image_path):
    try:
        if overlay_image_path.endswith('.tiff') or overlay_image_path.endswith('.tif'):
            # Using tifffile library
            overlay_image = imread(overlay_image_path)
        else:
            overlay_image = cv2.imread(
                overlay_image_path, cv2.IMREAD_UNCHANGED)

        if overlay_image.shape[-1] == 4:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2BGR)

        return overlay_image
    except Exception as e:
        print(f"Error reading overlay image: {e}")
        return None


def update_color_mask(df, segmentation_mask, color_mask, selected_channel):
    """
    Updates the color mask based on DataFrame values for a selected channel.

    Parameters:
    - df: DataFrame containing the cell data
    - segmentation_mask: numpy array representing the segmentation mask
    - color_mask: numpy array to be updated with new colors
    - selected_channel: str, the channel for which to update the color mask
    """
    cmap = plt.cm.turbo_r

    vmin, vmax = df[selected_channel].min(), df[selected_channel].max()
    norm = plt.Normalize(vmin, vmax)

    # Convert DataFrame values to RGB color values
    data_colors = cmap(norm(df[selected_channel].values))
    df['color_values'] = [tuple(int(v * 255)
                                for v in color[:3]) for color in data_colors]

    # Map cell IDs in the segmentation mask to DataFrame indices
    cell_positions = {}
    for index, row in df.iterrows():
        center_x, center_y = int(row['Location_Center_X']), int(
            row['Location_Center_Y'])
        cell_id = segmentation_mask[center_y, center_x]
        cell_positions[cell_id] = index

    # Update the color mask
    for cell_id, index in cell_positions.items():
        color = tuple(df.loc[index, 'color_values'])
        color_mask[segmentation_mask == cell_id] = color


class CSVProcessor:
    @staticmethod
    def read_csv(file_path):
        """
        Reads a CSV file and returns a Pandas DataFrame.
        Attempts to auto-detect the delimiter if reading fails.

        Parameters:
        - file_path: str, path to the CSV file

        Returns:
        - Pandas DataFrame if successful, None otherwise
        """
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                raise ValueError(f"Only one column found.")
        except Exception as e:
            print(f"WARNING: {e}. Trying to auto-detect delimiter.")
            detected_delimiter = CSVProcessor.detect_delimiter(file_path)

            if detected_delimiter:
                print(f"Attempting to read the file again using detected delimiter: '{detected_delimiter}'")
                df = pd.read_csv(file_path, delimiter=detected_delimiter)
            else:
                print("Could not auto-detect a common delimiter. Please check the file.")
                return None

        return df

    @staticmethod
    def detect_delimiter(file_path):
        """
        Auto-detects the delimiter in a CSV file.

        Parameters:
        - file_path: str, path to the CSV file

        Returns:
        - str representing the detected delimiter, or None if not found
        """
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()

        common_delimiters = [',', ';', '\t']

        for delim in common_delimiters:
            if delim in first_line:
                return delim
        return None

    @staticmethod
    def save_histograms(data, graph_path, csv_name, channels_to_histogram, suffix, cluster_number=None):
        """
        Saves histograms of specified columns in the data to a specified path.

        Parameters:
        - data: Pandas DataFrame, the data for which to generate histograms
        - graph_path: str, path to save the histogram graphs
        - csv_name: str, name of the CSV file (used in naming the graphs)
        - channels_to_histogram: list of str, channels for which to save histograms
        - suffix: str, a suffix to append to the filename
        - cluster_number: int, the cluster number (optional)
        """

        if cluster_number is not None:
            data = data[data['cluster'] == cluster_number]

        for column in channels_to_histogram:
            if column in data.columns:
                sns.displot(data, x=column)
                plt.title(f"Histogram for {column}")

                # Conditionally append the suffix, underscore, and cluster number
                filename_suffix = f"_{suffix}" if suffix else ''
                cluster_suffix = f"_cluster_{cluster_number}" if cluster_number is not None else ''
                save_path = os.path.join(graph_path, f"{csv_name}_{column}{filename_suffix}{cluster_suffix}_histogram.png")

                plt.tight_layout()
                plt.savefig(save_path, dpi=600)
                plt.close()

                # Conditionally include the suffix and cluster number in the log message
                log_suffix = f" with suffix '{suffix}'" if suffix else ''
                log_cluster = f" for cluster {cluster_number}" if cluster_number is not None else ''
                print(f"Saving histogram of {column} from csv file {csv_name}{log_suffix}{log_cluster}")
            else:
                print(f"WARNING: Column '{column}' not found in data. Skipping histogram.")

    @staticmethod
    def normalize_by_pixel_size(data, pixel_size, channels_to_normalize, area_column):
        """
        Normalizes data by pixel size.

        Parameters:
        - data: Pandas DataFrame, the data to normalize
        - pixel_size: float, pixel size for normalization
        - channels_to_normalize: list of str, columns to normalize
        - area_column: str, column representing the area

        Returns:
        - Pandas DataFrame with normalized data
        """
        normalized_data = data.copy()
        try:
            if area_column not in normalized_data.columns:
                print(f'WARNING: Area column "{area_column}" not found in data. Normalization will be skipped.')
                return normalized_data

            normalized_data[area_column] = normalized_data[area_column] * \
                (pixel_size ** 2)
            normalized_data[channels_to_normalize] = normalized_data[channels_to_normalize].div(
                normalized_data[area_column], axis=0)
        except Exception as e:
            print(f'WARNING: Normalization failed due to {e}. Using original data.')
        return normalized_data

    @staticmethod
    def remove_outliers_percentiles(df, columns, p_low=0.0, p_high=0.997):
        """
        Removes outliers based on percentiles.

        Parameters:
        - df: Pandas DataFrame, the data to filter
        - columns: list of str, columns to apply filtering
        - p_low: float, lower percentile for filtering. Defaults to 0.0.
        - p_high: float, upper percentile for filtering. Defaults to 0.997.

        Returns:
        - Pandas DataFrame with outliers removed
        """
        mask = pd.Series([True] * len(df))
        for col in columns:
            low = df[col].quantile(p_low)
            high = df[col].quantile(p_high)
            col_mask = (df[col] >= low) & (df[col] <= high)
            mask = mask & col_mask
        return df[mask].reset_index(drop=True)

    @staticmethod
    def remove_outliers_zscore(df, columns, n_std=3):
        """
        Removes outliers based on Z-score.

        Parameters:
        - df: Pandas DataFrame, the data to filter
        - columns: list of str, columns to apply filtering
        - n_std: int, number of standard deviations for filtering

        Returns:
        - Pandas DataFrame with outliers removed
        """
        mask = pd.Series([True] * len(df))
        for col in columns:
            mean = df[col].mean()
            sd = df[col].std()
            col_mask = df[col] <= mean + (n_std * sd)
            mask = mask & col_mask
        return df[mask].reset_index(drop=True)

    @staticmethod
    def save_dataframe(data, save_path, csv_name, suffix=''):
        """
        Saves the DataFrame to a CSV file with a given suffix.

        Parameters:
        - data: Pandas DataFrame, the data to save
        - save_path: str, the directory in which to save the file
        - csv_name: str, the base name of the CSV file
        - suffix: str, a suffix to append to the filename
        """
        full_path = os.path.join(save_path, f"{csv_name}_{suffix}.csv")
        data.to_csv(full_path, index=False)
        print(f"Saving DataFrame to {full_path}")


def main():
    """
    Main function to execute the script.
    """
    args = parse_arguments()

    # Identify CSV files in the specified folder
    csv_files = [f for f in os.listdir(args.folder) if f.endswith('.csv')]

    # Create or identify directory for saving graphs and CSVs
    graph_path = args.working_directory or args.folder
    graph_path = os.path.join(graph_path, 'output', 'graphs')
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    csv_path = args.working_directory or args.folder
    csv_path = os.path.join(csv_path, 'output', 'CSVs')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    # Initialize variables
    channels_to_normalize = []
    channels_to_filter = []
    channels_to_histogram = []
    channels_to_heatmap = []
    channels_to_cluster = []
    area_column = 'AreaShape_Area'

    # Read and interpret setup_csv if provided
    if args.setup_csv:
        setup_csv_path = args.setup_csv  # get the value from argparse
        channels_to_normalize, channels_to_filter, area_column, channels_to_histogram, channels_to_heatmap, channels_to_cluster, channels_to_cluster_heatmap = read_setup_csv(
            setup_csv_path, area_column)

    # Capture values from parsed arguments
    n_std = args.n_std

    try:
        p_low, p_high = map(float, args.percentiles.split(','))
    except ValueError:
        print("WARNING: Invalid format for --percentiles. Should be a comma-separated pair of floats. Using defaults 0.0 and 0.997.")
        p_low, p_high = 0.0, 0.997

    # Process each CSV file
    for csv_file in csv_files:
        data_path = os.path.join(args.folder, csv_file)

        segmentation_mask = None
        color_mask = None
        blended_image = None
        overlay_image = None
        df = CSVProcessor.read_csv(data_path)
        scaled_df = None

        can_generate_heatmap = True

        if df is None:
            print(f"WARNING: Skipping '{csv_file}' due to read failure.")
            continue

        # Apply normalization if required
        if args.pixel_size and channels_to_normalize:
            normalized_df = CSVProcessor.normalize_by_pixel_size(
                df, args.pixel_size, channels_to_normalize, area_column)
        else:
            print(
                "WARNING: No pixel size and or channels to normalise were specified. Skipping normalisation step.")

        preprocessed_df = normalized_df.copy()

        # Perform outlier filtering if a method is specified
        if args.outlier_filtering_method:

            if args.outlier_filtering_method == 'percentiles':
                preprocessed_df = CSVProcessor.remove_outliers_percentiles(
                    preprocessed_df, channels_to_filter, p_low, p_high)
            elif args.outlier_filtering_method == 'zscore':
                preprocessed_df = CSVProcessor.remove_outliers_zscore(
                    preprocessed_df, channels_to_filter, n_std)
            
            filtered_rows = df.index.difference(preprocessed_df.index)
            df = df.drop(filtered_rows)

        else:
            print(
                "WARNING: No outlier filtering method was specified. Skipping this step.")

        # Look for a corresponding segmentation mask
        segmentation_mask_name = f"{csv_file.split('.')[0]}_mask.png"
        segmentation_mask_path = args.folder
        segmentation_mask = read_segmentation_mask(
            segmentation_mask_path, segmentation_mask_name)

        if segmentation_mask is None:
            print(f"WARNING: No segmentation mask found for {csv_file}. Skipping heatmap generation.")
            can_generate_heatmap = False
        else:
            color_mask = np.zeros(
                (segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)

        # Look for a corresponding overlay image
        overlay_image_name = f"{csv_file.split('.')[0]}_overlay.tiff"
        overlay_image_path = os.path.join(args.folder, overlay_image_name)
        overlay_image = read_overlay_image(overlay_image_path)
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)
        if overlay_image is None:
            print(f"WARNING: No overlay image found for {csv_file}. Generating heatmap without overlay.")

        # Save overlay heatmaps of each marked channel
        if can_generate_heatmap:

            for selected_channel in channels_to_heatmap:
                if segmentation_mask is not None:
                    update_color_mask(df, segmentation_mask,
                                      color_mask, selected_channel)

                overlay_image = overlay_image.astype(color_mask.dtype)

                if overlay_image is not None and color_mask is not None:
                    blended_image = cv2.addWeighted(
                        color_mask, 0.6, overlay_image, 0.4, 0)
                elif color_mask is not None:
                    blended_image = color_mask

                if blended_image is not None:
                    height, width = blended_image.shape[:2]

                    orientation = 'horizontal' if width > height else 'vertical'

                    fig_height = height / 100.0
                    fig_width = width / 100.0

                    fig, ax1 = plt.subplots(
                        figsize=(fig_width, fig_height), dpi=600)

                    ax1.axis('off')

                    blended_image_rgb = cv2.cvtColor(
                        blended_image, cv2.COLOR_BGR2RGB)

                    im = ax1.imshow(blended_image_rgb)

                    divider = make_axes_locatable(ax1)

                    if orientation == 'horizontal':
                        padding = 0.2
                        cbar_size = "1%"
                    else:
                        padding = 0.5
                        cbar_size = "5%"

                    cax = divider.append_axes(
                        "right", size=cbar_size, pad=padding)

                    vmin, vmax = df[selected_channel].min(
                    ), df[selected_channel].max()
                    norm = plt.Normalize(vmin, vmax)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(
                        norm=norm, cmap='turbo'), cax=cax)
                    cbar.set_label(selected_channel, color='white', size=13)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    cbar.outline.set_edgecolor('white')
                    cbar.ax.tick_params(labelsize=13)
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                             color='white')

                    fig.patch.set_facecolor('black')
                    ax1.set_facecolor('black')
                    cax.set_facecolor('black')

                    # Save the figure
                    save_path = os.path.join(graph_path, f"{csv_file.split('.')[0]}_{selected_channel}_heatmap.png")
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))

                    plt.savefig(save_path, dpi=600, facecolor='black',
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close()

        # Save histograms of raw data if required
        if args.save_histograms and channels_to_histogram:
            CSVProcessor.save_histograms(df, graph_path, csv_file.split('.')[0],
                                         channels_to_histogram, '')

        # Save histograms of processed data if required
        if args.save_processed_histograms:
            suffix = ''
            if args.pixel_size and channels_to_normalize:
                suffix += 'normalised_'
            if args.outlier_filtering_method:
                suffix += 'filtered_'
            if suffix:
                CSVProcessor.save_histograms(preprocessed_df, graph_path, csv_file.split('.')[
                                             0], channels_to_histogram, suffix[:-1])

        # Save processed DataFrame as CSV if required
        if args.save_processed_csv:
            suffix = ''
            if args.pixel_size and channels_to_normalize:
                suffix += 'normalised_'
            if args.outlier_filtering_method:
                suffix += 'filtered_'
            if suffix:
                CSVProcessor.save_dataframe(
                    preprocessed_df, csv_path, csv_file.split('.')[0], suffix[:-1])

        if not args.no_cluster:

            try:
                k, resolution_parameter, seed = map(
                    str, args.clustering_parameters.split(','))
                print("Using the following clustering parameters: \nk = " + k + 
                    "\nresolution_parameter = " + resolution_parameter + 
                    "\nseed = " + seed)

                k = int(k)

                resolution_parameter = float(resolution_parameter)

                seed = None if seed == '' else int(seed)
            except ValueError:
                print("Invalid format for --clustering_parameters. It should be a comma-separated list of three values: 'k,resolution_parameter,seed'")

            # Perform scaling based on user choice
            if args.scaling_method == 'robust':
                r_scaler = RobustScaler()
                r_scaled = r_scaler.fit_transform(
                    preprocessed_df.loc[:, channels_to_cluster])
                scaled_df = pd.DataFrame(r_scaled, columns=channels_to_cluster)

            elif args.scaling_method == 'minmax':
                mm_scaler = MinMaxScaler()
                mm_scaled = mm_scaler.fit_transform(
                    preprocessed_df.loc[:, channels_to_cluster])
                scaled_df = pd.DataFrame(
                    mm_scaled, columns=channels_to_cluster)

            # Phenograph clustering
            communities, graph, Q = phenograph.cluster(
                scaled_df,
                clustering_algo="leiden",
                k=k,
                resolution_parameter=resolution_parameter,
                seed=seed,
            )

            preprocessed_df["cluster"] = pd.Categorical(communities)
            df["cluster"] = pd.Categorical(communities)

            # Count the size of each cluster and sort them
            cluster_counts = preprocessed_df['cluster'].value_counts(
            ).sort_values(ascending=False)

            print(str(max(preprocessed_df["cluster"]) + 1) +
                  " clusters have been detected.")
            print("Cluster sizes (in descending order):")
            print(cluster_counts)

            # Save histograms of raw data if required
            if args.save_histograms and channels_to_histogram:
                for cluster_number in df['cluster'].unique():
                    CSVProcessor.save_histograms(df, graph_path, csv_file.split('.')[0],
                                                 channels_to_histogram, 'raw', cluster_number)

            # Save histograms for individual clusters
            if args.save_processed_histograms:
                for cluster_number in preprocessed_df['cluster'].unique():
                    CSVProcessor.save_histograms(preprocessed_df, graph_path, csv_file.split('.')[0],
                                                 channels_to_histogram, suffix[:-1], cluster_number)

            # Save overlay heatmaps for each cluster
            if not args.no_cluster and channels_to_heatmap:
                color_mask = np.zeros(
                    (segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)

                for cluster_number in df['cluster'].unique():
                    print(f"Saving overlay heatmap for cluster: {cluster_number}")

                    # Reset color_mask for each cluster
                    color_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3),
                                           dtype=np.uint8)

                    df_cluster = df[df['cluster']
                                                 == cluster_number].copy()

                    for selected_channel in channels_to_heatmap:
                        if segmentation_mask is not None:
                            update_color_mask(df_cluster, segmentation_mask,
                                              color_mask, selected_channel)

                        # Check and resize overlay_image if necessary
                        overlay_image = overlay_image.astype(color_mask.dtype)

                        if overlay_image is not None and color_mask is not None:
                            blended_image = cv2.addWeighted(
                                color_mask, 0.6, overlay_image, 0.4, 0)
                        elif color_mask is not None:
                            blended_image = color_mask

                        if blended_image is not None:
                            height, width = blended_image.shape[:2]

                            orientation = 'horizontal' if width > height else 'vertical'

                            fig_height = height / 100.0
                            fig_width = width / 100.0

                            fig, ax1 = plt.subplots(
                                figsize=(fig_width, fig_height), dpi=600)

                            ax1.axis('off')

                            blended_image_rgb = cv2.cvtColor(
                                blended_image, cv2.COLOR_BGR2RGB)

                            im = ax1.imshow(blended_image_rgb)

                            divider = make_axes_locatable(ax1)

                            if orientation == 'horizontal':
                                padding = 0.2
                                cbar_size = "1%"
                            else:
                                padding = 0.5
                                cbar_size = "5%"

                            cax = divider.append_axes(
                                "right", size=cbar_size, pad=padding)

                            vmin, vmax = df_cluster[selected_channel].min(
                            ), df_cluster[selected_channel].max()
                            norm = plt.Normalize(vmin, vmax)
                            cbar = plt.colorbar(plt.cm.ScalarMappable(
                                norm=norm, cmap='turbo'), cax=cax)
                            cbar.set_label(selected_channel,
                                           color='white', size=13)
                            cbar.ax.yaxis.set_tick_params(color='white')
                            cbar.outline.set_edgecolor('white')
                            cbar.ax.tick_params(labelsize=13)
                            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'),
                                     color='white')

                            fig.patch.set_facecolor('black')
                            ax1.set_facecolor('black')
                            cax.set_facecolor('black')

                            save_path = os.path.join(graph_path, f"{csv_file.split('.')[0]}_{selected_channel}_cluster_{cluster_number}_heatmap.png")
                            plt.savefig(save_path, dpi=600, facecolor='black',
                                bbox_inches='tight', pad_inches=0.1)
                            plt.close()

        elif args.no_cluster and not args.no_umap:
            # Perform scaling based on user choice
            if args.scaling_method == 'robust':
                r_scaler = RobustScaler()
                r_scaled = r_scaler.fit_transform(
                    preprocessed_df.loc[:, channels_to_cluster])
                scaled_df = pd.DataFrame(r_scaled, columns=channels_to_cluster)

            elif args.scaling_method == 'minmax':
                mm_scaler = MinMaxScaler()
                mm_scaled = mm_scaler.fit_transform(
                    preprocessed_df.loc[:, channels_to_cluster])
                scaled_df = pd.DataFrame(
                    mm_scaled, columns=channels_to_cluster)

        else:
            print("Clustering disabled by --no_clustering flag. No clustering performed.")

        # UMAP Dimensionality Reduction
        if not args.no_umap:
            try:
                n_neighbors, min_dist, random_state = map(
                    str, args.umap_parameters.split(','))

                n_neighbors = int(n_neighbors)

                min_dist = float(min_dist)

                random_state = None if random_state == '' else int(
                    random_state)
            except ValueError:
                print("Invalid format for --umap_parameters. It should be a comma-separated list of three values: 'n_neighbors,min_dist,random_state'")

            reducer = umap.UMAP(n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=random_state)
            embedding = reducer.fit_transform(scaled_df)

            embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])

            if not args.no_cluster:
                embedding_df["cluster"] = preprocessed_df["cluster"]

                N = max(embedding_df["cluster"]) + 1

                existing_cmap = plt.get_cmap("jet")
                colors = existing_cmap(np.linspace(0, 1, N))

                new_cmap = mcolors.ListedColormap(colors)

                plt.figure(figsize=(10, 8))
                plt.scatter(embedding_df["UMAP1"], embedding_df["UMAP2"],
                            c=embedding_df["cluster"], cmap=new_cmap, vmin=0, vmax=N-1)

                cbar = plt.colorbar(ticks=np.arange(0, N))
                cbar.set_label('Cluster ID')

                plt.title('UMAP plot colored by cluster')
                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')

                umap_plot_path = os.path.join(graph_path, f"{csv_file.split('.')[0]}_UMAP_plot.png")
                plt.savefig(umap_plot_path, dpi=600)
                plt.close()

            else:
                plt.figure(figsize=(10, 8))
                plt.scatter(embedding_df["UMAP1"],
                            embedding_df["UMAP2"])
                plt.title('UMAP plot colored by cluster')
                plt.xlabel('UMAP1')
                plt.ylabel('UMAP2')

                umap_plot_path = os.path.join(graph_path, f"{csv_file.split('.')[0]}_UMAP_plot.png")
                plt.savefig(umap_plot_path, dpi=600)
                plt.close()

        else:
            print(
                "UMAP disabled by --no_umap flag. No dimensionality reduction performed.")

        # Generate cluster heatmaps and save clusters as individual/combined csv
        if not args.no_cluster:
            # Aggregate and scale clusters for cluster heatmap
            aggregation_method = args.aggregation_method

            cluster_list = [preprocessed_df[preprocessed_df["cluster"] == c]
                            for c in preprocessed_df["cluster"].unique()]

            aggregated_clusters = pd.DataFrame()

            for cluster_df in cluster_list:
                if aggregation_method == "median":
                    aggregated_value = cluster_df.median(
                        axis=0, numeric_only=True)
                else:  # mean
                    aggregated_value = cluster_df.mean(
                        axis=0, numeric_only=True)

                aggregated_clusters = pd.concat(
                    [
                        aggregated_clusters,
                        pd.DataFrame(aggregated_value).transpose(),
                    ],
                    ignore_index=True,
                )

            # Applies zscore scaling to the aggregated values for better heatmap representation
            z_scaler = StandardScaler()
            z_scaled = z_scaler.fit_transform(aggregated_clusters)
            heatmap_clusters = pd.DataFrame(
                z_scaled, columns=aggregated_clusters.columns)

            # Generate cluster heatmap
            if channels_to_cluster_heatmap:
                scaled_cluster_data = heatmap_clusters.loc[:,
                                                           channels_to_cluster_heatmap]
                cluster_sizes = [len(cluster_df)
                                 for cluster_df in cluster_list]

                sorted_cluster_sizes = sorted(cluster_sizes, reverse=True)

                cmap = sns.diverging_palette(255, 12, as_cmap=True)

                fig, ax = plt.subplots(figsize=(13, 10))

                ax = sns.heatmap(
                    scaled_cluster_data,
                    cmap=cmap,
                    annot=False,
                    linewidths=0.5,
                    linecolor="black",
                    cbar_kws={"pad": 0.15},
                    center=0.0,
                )

                ax2 = ax.twinx()

                sns.heatmap(
                    scaled_cluster_data,
                    cmap=cmap,
                    annot=False,
                    linewidths=0.5,
                    linecolor="black",
                    ax=ax2,
                    cbar=False,
                    center=0.0,
                )

                xtick_labels = [label for label in channels_to_cluster_heatmap]

                ax.set_xticks([*np.arange(0.5, len(scaled_cluster_data.columns), 1)])
                ax.set_xticklabels(xtick_labels, rotation=45,
                                   ha="right", rotation_mode="anchor")
                ax.tick_params(axis="y", rotation=0)

                ax2.set_yticklabels(sorted_cluster_sizes, rotation=0)
                ax2.tick_params(axis="y", rotation=0)
                ax2.set(ylabel="cell counts per cluster")

                fig.tight_layout()

                # Save the heatmap as a PNG file
                heatmap_path = os.path.join(graph_path, f"{csv_file.split('.')[0]}_cluster_heatmap.png")
                fig.savefig(heatmap_path, dpi=600)
                plt.close(fig)

            # Save individual cluster data
            if args.save_individual_clusters:
                for idx, cluster_df in enumerate(cluster_list):
                    individual_csv_path = os.path.join(csv_path, f"{csv_file.split('.')[0]}_cluster_{idx}.csv")

                    cluster_df.to_csv(individual_csv_path, index=False)

            # Save combined cluster data
            if args.save_combined_clusters:
                combined_csv_path = os.path.join(csv_path, f"{csv_file.split('.')[0]}_combined_clusters.csv")

                if "cluster" in preprocessed_df.columns:
                    preprocessed_df['cluster'] = preprocessed_df['cluster']

                preprocessed_df.to_csv(combined_csv_path, index=False)

        else:
            print(
                "Clustering disabled by --no_clustering flag. No cluster heatmap created.")


if __name__ == "__main__":
    main()
