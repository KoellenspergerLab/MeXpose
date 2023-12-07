# Pixel Script Manual

1. [Overview](#overview)
2. [Setup CSV File](#setup-csv-file)
3. [Command-Line Arguments](#command-line-arguments)
4. [Example Usage](#example-usage)
5. [Output](#output)
6. [Troubleshooting](#troubleshooting)

## Overview

This command-line script performs operations on pixel data, including k-means clustering and correlation heatmaps. The clustering assigns each pixel event to a specific cluster, and the heatmap helps identify correlations between different channels.

## Setup CSV File

The `setup.csv` file is used to configure how the script operates on data. It specifies which channels from the data should be included in the clustering and heatmap generation steps. Here's how to structure the `setup.csv` file:

- The first column, labeled `channels`, should contain the names of the desired channels within the data.
- The second column, labeled `cluster`, should contain a 1 if the corresponding channel should be included in clustering, and a 0 otherwise.
- The third column, labeled `heatmap`, should contain a 1 if the corresponding channel should be included in the heatmap, and a 0 otherwise.

### Example setup.csv

| channels                                      	| cluster | heatmap |
| ------------------------------------------------ 	| ------- | ------- |
| Intensity_IntegratedIntensity_iron_filtered_c1 	| 1       | 1       |
| Intensity_IntegratedIntensity_cobalt_filtered_c1  | 0       | 1       |
| Intensity_IntegratedIntensity_nuclei_filtered_c1  | 1       | 0       |

This setup would include the `Intensity_IntegratedIntensity_iron_filtered_c1` channel in both the clustering and heatmap steps. The `Intensity_IntegratedIntensity_cobalt_filtered_c1` channel would only be included in the heatmap, and the `Intensity_IntegratedIntensity_nickel_filtered_c1` would only be included in the clustering.


## Command-Line Arguments

### Required Arguments

- `--data_csv`: Specifies the path to the data CSV file.

  **Linux/Mac**: `--data_csv="/path/to/data.csv"`

  **Windows**: `--data_csv="C:\path\to\data.csv"`

- `--setup_csv`: Specifies the path to the setup CSV file.

  **Linux/Mac**: `--setup_csv="/path/to/setup.csv"`

  **Windows**: `--setup_csv="C:\path\to\setup.csv"`

- `--image_dims`: Specifies the dimensions for the generated images in the format `width,height`.

  **Example**: `--image_dims=200,200`

### Optional Arguments

- `--no_cluster`: Skip the clustering step if specified.

  **Example**: `--no_cluster`

- `--no_heatmap`: Skip the heatmap generation step if specified.

  **Example**: `--no_heatmap`

- `--n_clusters`: Specifies the number of clusters for KMeans (default is 3).

  **Example**: `--n_clusters=5`

- `--no_transform`: Disables square root transformation of data.

  **Example**: `--no_transform`
  
- `--scaling_method`: Choose the scaling method for clustering. Options are `robust` and `minmax`. Default is `robust`.

  **Example**: `--scaling_method=minmax`

- `--correlation_method`: Select the desired correlation method for the heatmap. Options are `pearson` and `spearman`. Default is `spearman`.

  **Example**: `--correlation_method=pearson`

## Example Usage

#### Linux/MacOS:

```
python pixel_script.py --data_csv="/path/to/data.csv" \
                       --setup_csv="/path/to/setup.csv" \
                       --image_dims="100,100" \
                       --no_cluster \
                       --no_heatmap \
                       --n_clusters=4
```

#### Windows:

```
python pixel_script.py --data_csv="C:\path\to\data.csv" ^
                       --setup_csv="C:\path\to\setup.csv" ^
                       --image_dims="100,100" ^
                       --no_cluster ^
                       --no_heatmap ^
                       --n_clusters=4
```

Replace placeholders like `path/to/csv/data.csv` or `C:\path\to\data.csv` with your actual paths and adjust the values as needed.

## Output

The script generates output in a subfolder named `output` within the directory containing the `data_csv` file. The output includes:

- `clustered_image.png`: An image where each pixel is colored according to its cluster.
- `correlation_heatmap.png`: A heatmap indicating the correlation between different channels.
- `statistics.txt`: A text file containing statistics related to the clusters.

## Troubleshooting

### KeyError or IndexError

If you encounter a KeyError or IndexError, make sure that the channels specified in `setup.csv` exist in `data_csv`.

