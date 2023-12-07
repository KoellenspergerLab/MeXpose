# Quantification Notebook

This wiki aims to briefly explain the functions and or requirements for each cell of the Quantification Notebook. Note that the cell numbers in this wiki must not be confused with the execution count (in brackets left of a cell) which indicates the cell's position in the execution order.

This notebook expects a 'standards' CSV file which contains two (in future three) columns in the following order without a column header:

Names of the standards (e.g. isotope names) - string  
The factor by which to divide for quantification (e.g. slope of calibration curve) - float  
(Not yet supported) The y-axis section of a calibration curve - float

### Cell 1: Import Libraries and Set Up Environment

#### Functionality
This cell imports all the essential Python libraries required for the notebook to function. These libraries assist in data manipulation, data visualization, and interactive widgets.

#### Libraries Imported
- **Standard Libraries**: `io`, `os` for handling basic I/O and file operations.
- **Data Manipulation**: `pandas` (as `pd`) and `numpy` (as `np`) for data manipulation and mathematical operations.
- **Data Visualization**: `matplotlib.pyplot` (as `plt`) and `seaborn` (as `sns`) for plotting and data visualization.
- **Interactive Widgets**: `ipywidgets` (imported as `SelectMultiple`, `TwoByTwoLayout`, `interact`, `widgets`) for creating interactive user interfaces.

---

### Cell 2: File Upload and Path Configuration

#### Functionality
- Provides widgets to upload single-cell data files and standard files in CSV format.
- Allows the user to specify a working directory or desired save path.

#### Widgets
- `singlecell_files`: A file upload widget that accepts multiple CSV files containing single-cell data.
- `standard_file`: A file upload widget that accepts a single CSV file containing standard values.
- `save_path_widget`: A text input widget where the user can specify a working directory or desired save path.

### Cell 3: Data Reading and Formatting

#### Functionality
Reads and formats both single-cell and standard files uploaded by the user.

#### Code Explanation
1. Reads each uploaded single-cell file into a DataFrame and stores them in a list called `singlecell_dfs`.
2. Reads the uploaded standard file into a DataFrame called `standards_df`.

##### Additional Notes:
Assumes that you upload only one standard file and reads it into a DataFrame. This DataFrame is then transposed and formatted.

---

### Cell 4: Select Channels and Standard Isotopes

#### Functionality
- Provides widgets to select channels and standard isotopes for quantification.
- Allows the user to make multiple selections.

#### Widgets
- `quant_channels`: Allows the user to select multiple channels for quantification.
- `std_isotopes`: Allows the user to select multiple standard isotopes for quantification.
  
#### Code Explanation
- Initializes two lists, `quant_channels_selected` and `std_isotopes_selected`, to hold the selected options.
- Attaches change listeners to the widgets. These listeners update the selected options when the user interacts with the widgets.

#### Additional Notes:
Make sure you select channel and corresponding isotopes in the same order otherwise you will obtain false results

### Cell 5: Quantify Selected Isotopes and Plot Histograms

#### Functionality
Quantifies the selected isotopes based on the standards and plots the quantified distributions.

#### Code Explanation
1. Copies the single-cell DataFrames to a list called `quant_df`.
2. Performs quantification on the selected channels by dividing by the corresponding standard isotopes.
3. Creates subplots for each selected channel and file, then fills them with histograms showing the quantified distribution.

---

### Cell 6: Save Individual Histograms

#### Functionality
Saves histograms of the quantified distributions for each selected channel and single-cell file.

#### Code Explanation
- Generates a Seaborn `displot` for each quantified distribution.
- Saves the plot as a PNG file in the directory specified in `save_path_widget`.

### Cell 7: Save Quantified Data as CSV Files

#### Functionality
- Saves the quantified single-cell data as CSV files.

#### Code Explanation
1. Makes a copy of each DataFrame in `quant_df`.
2. Appends "_quantified" to the column names selected by `quant_channels`.
3. Saves the modified DataFrame as a CSV file in the directory specified in `save_path_widget`.
