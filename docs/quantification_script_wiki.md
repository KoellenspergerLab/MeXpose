# Quantification Script

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Command-Line Flags](#command-line-flags)
4. [Configuration Files](#configuration-files)
5. [Examples](#examples)

---

## Overview

The Quantification Script is designed to process and quantify single-cell data. It allows you to apply various transformations and save histograms or quantified data as CSV files. The scripts expects two files as input, a 'setup.csv' file and a 'standards.csv' file. This manual guides you through the scripts usage and the expected content of the files.

---

## Getting Started

#### Linux/Mac

To run the script, navigate to the directory containing the script or enter its absolute location and execute the following command:

```bash
python quantification_script.py \
--data_directory path/to/data/ \
--standards_csv path/to/standards.csv \
--setup_csv path/to/setup.csv
```

#### Windows

```cmd
python quantification_script.py ^
--data_directory path\to\data\ ^
--standards_csv path\to\standards.csv ^
--setup_csv path\to\setup.csv
```

### Important Notes

- All the `.csv` files to be analyzed should be in the `--data_directory`.
- The `--standards_csv` and `--setup_csv` files should be formatted as described in the [Configuration Files](#configuration-files) section.

---

## Command-Line Flags

### `--data_directory`

- **Purpose**: Specifies the directory containing the `.csv` files to be processed.
- **Example**: `--data_directory path/to/data/`  (Linux) or `--folder path\to\data` (Windows)
- **Default**: None

### `--standards_csv`

- **Purpose**: Specifies the path to the `standards.csv` file.
- **Example**: `--standards_csv path/to/standards.csv` (Linux) or `--folder path\to\standards.csv` (Windows)
- **Default**: None

### `--setup_csv`

- **Purpose**: Specifies the path to the `setup.csv` file.
- **Example**: `--setup_csv path/to/setup.csv`  (Linux) or `--folder path\to\setup.csv` (Windows)
- **Default**: None

### `--save_histograms`

- **Purpose**: When set, saves histograms for each channel specified in `setup.csv`.
- **Example**: `--save_histograms`
- **Default**: False

### `--save_channel_data`

- **Purpose**: When set, saves quantified data for each channel as a separate .csv file.
- **Example**: `--save_channel_data`
- **Default**: False

### `--save_combined_data`

- **Purpose**: When set, saves all quantified data into a single .csv file.
- **Example**: `--save_combined_data`
- **Default**: False

---

## Configuration Files

### `setup.csv`

This file specifies which channels should be quantified and whether histograms should be created for each channel.

| channel       | standard    | quantify | histogram |
|---------------|-------------|----------|-----------|
| Channel\_1    | Standard\_1 | 1        | 1         |
| Channel\_2    | Standard\_2 | 1        | 1         |
| Channel\_3    | Standard\_3 | 0        | 0         |

- **channel**: The name of the channel.
- **standard**: The name of the standard used for quantification. This should match an entry in `standards.csv`.
- **quantify**: 1 for 'Yes' and 0 for 'No'.
- **histogram**: 1 for 'Yes' and 0 for 'No'.

### `standards.csv`

This file contains the standards used for quantifying the channels, specifying the slope and y-axis values.

| standard     | slope  | y-axis |
|--------------|--------|--------|
| Standard_1   | 1.52   | 11.69  |
| Standard_2   | 2.68   | 0.70   |

- **standard**: The name of the standard.
- **slope**: The factor by which to divide for quantification.
- **y-axis**: This field is currently not used but will be integrated in a future update.

---

## Examples

To run the script and save histograms:

#### For Linux/Mac

```bash
python quantification_script.py \
--data_directory path/to/data/ \
--standards_csv path/to/standards.csv \
--setup_csv path/to/setup.csv \
--save_histograms
```

#### For Windows

```cmd
python quantification_script.py ^
--data_directory path\to\data\ ^
--standards_csv path\to\standards.csv ^
--setup_csv path\to\setup.csv ^
--save_histograms
```

Replace placeholders like `path/to/data` or `C:\path\to\data` with your actual paths and adjust the values as needed.

---
