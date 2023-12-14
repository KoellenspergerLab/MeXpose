# Analysis Script

## Introduction

This script is a comprehensive tool designed to facilitate the analysis of single-cell data, particularly in the context of multiplexed imaging. It performs multiple tasks such as data normalization, outlier removal, clustering, and various types of data visualization including histograms and heatmaps. The script is designed to be scalable, capable of handling large datasets efficiently.

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Command-Line Flags](#command-line-flags)
4. [Demo Commands](#demo-commands)

---

## Getting Started

To run the script, enter the scripts absolute location and execute the following command:

#### Linux/Mac
```
python path/to/analysis_script.py --folder path/to/folder
```

Replace `path/to/analysis_script.py` with the path of the script, and `path/to/folder/` with the directory containing your `.csv` files.

#### Windows
```
python analysis_script.py --folder X:\path\to\folder
```

Replace `X:\path\to\analysis_script.py` with the path of the script, and `X:\path\to\folder` with the directory containing your `.csv` files.

### Important Notes

- The folder specified should contain all the `.csv` files to be analyzed.
- If a `setup.csv` file is provided, it must be correctly formatted to specify how each channel should be processed.

---

## File Naming Scheme

Proper file naming is essential for the script to correctly identify and process data files, overlay images, and segmentation masks. Below are guidelines for each type of file:

### Data Files

- Data files should be in `.csv` format.
- The file name should not contain any spaces or special characters.
  
  **Example**: `Sample_Data_1.csv`, `Experiment2.csv`

### Overlay Images

- Overlay images should be in a format compatible with OpenCV (e.g., `.png`, `.jpg`, `.tiff`).
- The file name should start with the same name as its corresponding data file, followed by a specific identifier for overlay images.
  
  **Example**: For a data file named `Sample_Data_1.csv`, the corresponding overlay image could be named `Sample_Data_1_overlay.png`.

### Segmentation Masks

- Segmentation masks should be in `.png` format, compatible with OpenCV.
- The file name should start with the same name as its corresponding data file, followed by a specific identifier for segmentation masks.

  **Example**: For a data file named `Sample_Data_1.csv`, the corresponding segmentation mask could be named `Sample_Data_1_mask.png`.

By following these naming conventions, you help the script to automatically associate data files with their corresponding overlay images and segmentation masks, thereby streamlining the entire analysis process.

---

## The `setup.csv` File

### Overview

The `setup.csv` file is a configuration file that allows you to specify how each channel should be processed. This is particularly useful if you have multiple channels and want to apply different operations to them.

### File Structure

- `channels`: The name of the channel.
- `normalize`: Numeric value indicating whether to normalize this channel.
- `isArea`: Numeric value indicating whether this channel represents the volumetric area measure.
- `filter`: Numeric value indicating whether to filter this channel for outliers.
- `histogram`: Numeric value indicating whether to generate histograms for this channel.
- `heatmap`: Numeric value indicating whether to generate heatmaps for this channel.
- `cluster`: Numeric value indicating whether to include this channel in clustering.
- `cluster heatmap`: Numeric value indicating whether to generate cluster heatmaps for this channel.

For Numeric values, use 1 for `True` and 0 for `False`. This applies to all columns besides `channels`.

### Example

| channels     | normalize | isArea | filter | histogram | heatmap | cluster | cluster heatmap |
|--------------|-----------|--------|--------|-----------|---------|---------|-----------------|
| Channel1     | 1         | 0      | 1      | 1         | 1       | 1       | 1               |
| Channel2     | 1         | 0      | 0      | 0         | 0       | 1       | 0               |
| Area_Channel | 0         | 1      | 0      | 1         | 0       | 0       | 0               |


In this example:

- `Channel1` will be normalized, filtered for outliers, used for clustering, and used for histogram, heatmap, and cluster heatmap visualizations. It is not considered the Area channel.
- `Channel2` will be normalized but not filtered for outliers. It will be used for clustering but no histogram or heatmap will be generated.
- `Area_Channel` will not be normalized or filtered for outliers. It is considered the Area channel and a histogram will be created for it, but it will not be used for clustering and heatmap visualizations. It will be used for the cluster heatmap visialization.

Make sure to save this file in the same directory as your `.csv` data files or specify its path using the `--setup_csv` flag.

### How to Use

To use the `setup.csv` file, place it in the directory specified by the `--folder` flag, or specify its path directly using the `--setup_csv` flag:

#### Linux/Mac
```
# Example usage with --setup_csv flag
python script_name.py --folder path/to/csv/files --setup_csv path/to/setup.csv
```

#### Windows
```
# Example usage with --setup_csv flag
python script_name.py --folder X:\path\to\csv\files --setup_csv X:\path\to\setup.csv
```

---

## Command-Line Flags

Each flag serves a specific purpose and allows you to customize the behavior of the script. Below is a detailed explanation of each.

### `--folder` or `-f`

- **Purpose**: Specifies the directory containing the `.csv` files to be analyzed.
- **Example**: `--folder path/to/csv/files` (Linux) or `--folder path\to\csv\files` (Windows)
- **Default**: None

### `--working_directory` or `-wd`

- **Purpose**: Sets the directory where output files will be saved.
- **Example**: `--working_directory path/to/output` (Linux) or `--folder path\to\output` (Windows)
- **Default**: The folder containing the input files.

### `--setup_csv`

- **Purpose**: Path to the `setup.csv` file, which controls how each channel is processed.
- **Example**: `--setup_csv path/to/setup.csv` (Linux) or `--folder path\to\setup.` (Windows)
- **Default**: None

### `--pixel_size`

- **Purpose**: Sets the pixel size for normalization.
- **Example**: `--pixel_size 1.0`
- **Default**: None (Normalization will be skipped)

### `--outlier_filtering_method`

- **Purpose**: Specifies the method for outlier filtering.
- **Options**: `percentiles`, `zscore`
- **Example**: `--outlier_filtering_method percentiles`
- **Default**: None (Outlier filtering will be skipped)

### `--n_std`

- **Purpose**: Sets the number of standard deviations for Z-score filtering.
- **Example**: `--n_std 3`
- **Default**: `3`
- **Note**: Applicable only if `--outlier_filtering_method` is set to 'zscore'.

### `--percentiles`

- **Purpose**: Sets the lower and upper percentile values for percentile-based filtering.
- **Example**: `--percentiles 0.0,0.997`
- **Default**: `0.0,0.997`
- **Note**: Applicable only if `--outlier_filtering_method` is set to 'percentiles'.

### `--scaling_method`

- **Purpose**: Specifies the scaling method for used to scale data for clustering.
- **Options**: `robust`, `minmax`
- **Example**: `--scaling_method robust`
- **Default**: `robust`

### `--clustering_parameters`

- **Purpose**: Sets the parameters used for Phenograph clustering.
- **Options**: k, resolution_parameter, seed
- **Example**: `--clustering_parameters `
- **Default**: `30,1.0,42`

### `--no_cluster`

- **Purpose**: Disables clustering when set.
- **Example**: `--no_cluster`
- **Default**: False (Clustering enabled)
- **Note**: If set, cluster heatmaps will not be generated.

### `--umap_parameters`

- **Purpose**: Sets the parameters used for UMAP dimensionality reduction.
- **Options**: n\_neighbours, min\_dist, random\_state
- **Example**: `--umap_parameters 15,0.1,42`
- **Default**: `15,0.1,42`


### `--no_umap`

- **Purpose**: Disables UMAP dimensionality reduction when set.
- **Example**: `--no_umap`
- **Default**: False (UMAP enabled)

### `--aggregation_method`

- **Purpose**: Specifies the method used to aggregate data for each cluster in the cluster heatmaps.
- **Options**: `median`, `mean`
- **Example**: `--aggregation_method median`
- **Default**: `median`

### `--save_histograms`

- **Purpose**: Enables the saving of histograms for the specified channels.
- **Example**: `--save_histograms`
- **Default**: False

### `--save_processed_histograms`

- **Purpose**: Enables the saving of histograms after data preprocessing.
- **Example**: `--save_processed_histograms`
- **Default**: False

### `--save_processed_csv`

- **Purpose**: Enables the saving of processed DataFrames as CSV files.
- **Example**: `--save_processed_csv`
- **Default**: False

### `--save_individual_clusters`

- **Purpose**: Enables the saving of individual CSV files for each cluster.
- **Example**: `--save_individual_clusters`
- **Default**: False

### `--save_combined_clusters`

- **Purpose**: Enables the saving a combined CSV file for all clusters.
- **Example**: `--save_combined_clusters`
- **Default**: False

---

## Demo Commands

For quick testing or demonstration purposes, here's how you can run the script to generate the following output:

- size normalised data
- removing the top 0.3% of cells
- raw & processed histograms
- heatmap overlays
- processed CSVs

*Note*: You will need a setup.csv file for these commands.

#### Linux/Mac
```
python script_name.py \
  --folder /path/to/csv/files \
  --working_directory /path/to/output \
  --setup_csv /path/to/setup.csv \
  --pixel_size 1.0 \
  --outlier_filtering_method percentiles \
  --percentiles 0.0,0.997 \
  --no_cluster \
  --no_umap \
  --save_histograms \
  --save_processed_histograms \
  --save_processed_csv
```

#### Windows
```
python script_name.py ^
  --folder C:\path\to\csv\files ^
  --working_directory C:\path\to\output ^
  --setup_csv C:\path\to\setup.csv ^
  --pixel_size 1.0 ^
  --outlier_filtering_method percentiles ^
  --percentiles 0.0,0.997 ^
  --no_cluster ^
  --no_umap ^
  --save_histograms ^
  --save_processed_histograms ^
  --save_processed_csv
```

Replace placeholders like `path/to/csv/files` or `C:\path\to\csv\files` with your actual paths and adjust the values as needed.
