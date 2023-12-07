# Pixel Notebook

This wiki aims to briefly explain the functions and or requirements for each cell of the Pixel Notebook. Note that the cell numbers in this wiki must not be confused with the execution count (in brackets left of a cell) which indicates the cell's position in the execution order.

### Cell 1: Import Libraries and Set Up Environment

#### Functionality
This cell imports all the necessary Python libraries required for the notebook to function. These libraries support various operations like data manipulation, data visualization, clustering, and interactive widgets.

#### Libraries Imported
- **Standard Libraries**: `codecs`, `os` for basic Python operations.
- **Data Manipulation**: `pandas` (as `pd`) and `numpy` (as `np`) for data manipulation and mathematical operations.
- **Data Visualization**: `matplotlib` (as `mpl`), `matplotlib.pyplot` (as `plt`), and `seaborn` (as `sns`) for plotting and data visualization.
- **Interactive Widgets**: `ipywidgets` (as `widgets`), `IPython.display` for creating interactive user interfaces.
- **Clustering**: `KMeans` from `sklearn.cluster` for machine learning clustering tasks.
- **Data Preprocessing**: `RobustScaler`, `MinMaxScaler` from `sklearn.preprocessing` for data scaling.

#### Special Commands
- `%matplotlib widget`: This magic command enables the interactive Matplotlib backend for Jupyter notebooks.

#### Additional Notes
- Warning messages are optionally suppressed using the `warnings` library to make the notebook output cleaner.
- The `from io import StringIO` line imports the `StringIO` class for reading and writing strings as file streams.

---

### Cell 2: Define Widgets for Data Path and File Selection

#### Functionality
Initializes widgets for entering the working directory path and for selecting a CSV file from that directory. It also sets up a function to update the list of CSV files based on the given path.

#### Widgets
- **`image_path`**: Text widget for entering the working directory path.
    - **Type**: Text
    - **Placeholder**: "Enter your working directory path"
- **`csv_select`**: Radio buttons to select the CSV file containing single-cell data.
    - **Type**: Radio Buttons
    - **Options**: Populated based on the files in the provided directory path.

#### Function: `update_csv_files(change)`
Updates the list of available CSV files in the `csv_select` widget based on the directory path provided in `image_path`.

#### Output
Displays the `image_path` and `csv_select` widgets side by side in an HBox layout.

#### Additional Notes
The widgets are styled using a custom CSS style defined in the variable `style`.

### Cell 3: Load and Clean Single-Cell Data

#### Functionality
Reads the selected single-cell CSV data into a pandas DataFrame and drops any columns with missing values.

#### Variables
- **`PATH`**: Stores the value of the working directory path from `image_path`.
- **`PX_DATA`**: Stores the name of the selected CSV file from `csv_select`.
- **`file_path`**: Combines `PATH` and `PX_DATA` to form the complete file path.
- **`image_df`**: DataFrame holding the loaded and cleaned single-cell data.

#### Output
The cell doesn't have a visual output but it updates the `image_df` DataFrame with the loaded data.

#### Additional Notes
Columns with missing or `NaN` values are dropped from `image_df` using `dropna(axis=1)`.

---

### Cell 4: Configure Clustering Parameters and Data Scaling Method

#### Functionality
Sets up widgets for entering the number of clusters, selecting channels for clustering, and choosing the scaling method for data standardization.

#### Widgets
- **`number_clusters`**: Text box for entering the desired number of clusters.
    - **Type**: Text
    - **Placeholder**: "Desired number of clusters"
- **`channels_cluster`**: Multiple selection box for choosing which channels to use for clustering.
    - **Type**: Multiple Select
    - **Options**: Columns of the `image_df` DataFrame
- **`scaler_select`**: Radio buttons to choose the scaling method.
    - **Type**: Radio Buttons
    - **Options**: 'RobustScaler', 'MinMaxScaler'

#### Output
Displays the `number_clusters`, `channels_cluster`, and `scaler_select` widgets horizontally.

#### Additional Notes
For more information on data scaling methods, consult the [Scikit-learn documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) and [here](https://scikit-learn.org/stable/modules/preprocessing.html).

### Cell 5: Perform Clustering and Update DataFrame

#### Functionality
Performs KMeans clustering on the selected channels after scaling the data using the selected scaling method. The clustering labels are then added to the `image_df` DataFrame.

#### Variables
- **`cluster_df`**: A subset of `image_df` containing only the selected channels.
- **`cluster_std`**: The scaled version of `cluster_df`.
- **`kmeans`**: KMeans object from scikit-learn.

#### Functions
- **Scaling**: Uses either `RobustScaler` or `MinMaxScaler` based on user selection.
- **KMeans Clustering**: Uses the scikit-learn `KMeans` algorithm.

#### Output
Prints a message indicating that clustering has been completed.

#### Additional Notes
If only one channel is selected, `cluster_df` is reshaped to be 2D, as required by scikit-learn's scaling and clustering functions.

---

### Cell 6: Input Image Dimensions

#### Functionality
Sets up widgets for entering the dimensions of the image, specifically its width and height in pixels.

#### Widgets
- **`img_width`**: Text box for entering the image width in pixels.
    - **Type**: Text
    - **Placeholder**: "Enter your image width in pixels"
- **`img_height`**: Text box for entering the image height in pixels.
    - **Type**: Text
    - **Placeholder**: "Enter your image height in pixels"

#### Output
Displays the `img_width` and `img_height` widgets horizontally.

### Cell 7: Generate and Display Clustered Image

#### Functionality
Reshapes the clustering labels to match the original image dimensions and then displays this image.

#### Variables
- **`labels`**: Array containing the KMeans cluster labels extracted from `image_df`.
- **`image`**: 2D array formed by reshaping the `labels` array, representing the clustered image.

#### Output
Displays the clustered image in a 10x10 figure.

#### Additional Notes
- The `labels` array is reshaped based on the user-provided image dimensions (`img_width` and `img_height`).
- The Matplotlib parameter `savefig.pad_inches` is set to 0 to remove padding around the saved figure.

### Cell 8: Save Clustered Image (Optional)

#### Functionality
Contains code for saving the generated clustered image as a PNG file.

#### Output
No output unless the code is uncommented and executed.

#### Additional Notes
To save the clustered image, uncomment the line and execute the cell.

---

### Cell 9: Generate Cluster Statistics

#### Functionality
Calculates and prints the average, minimum, and maximum pixel values for each selected channel in each cluster. This information is useful for understanding the distribution of pixel values within each cluster.

#### Function: `generate_info(cluster_label, channel, df)`
- Calculates the mean, minimum, and maximum pixel values for a given channel in a specified cluster.
- Returns a string containing these statistics.

#### Variables
- **`info_strings`**: List to store all the generated information strings for each cluster and channel.

#### Output
Prints the statistics for each channel in each cluster. 

#### Additional Notes
This cell uses the `labels` array generated from previous cells and iterates over the range of unique cluster labels. For each cluster and channel, it calls the `generate_info` function to compute and print the statistics.

### Cell 10: Save Cluster Statistics to File (Optional)

#### Functionality
Contains code for saving the calculated statistics to a text file.

#### Output
No output unless the code is uncommented and executed.

#### Additional Notes
To save the statistics, uncomment the lines and execute the cell. The statistics will be saved in a text file named `kmeans_cluster_statistics.txt` in the specified path.

---

### Cell 11: Channel Selection for Correlation Analysis

#### Functionality
Initiates widgets for selecting the channels and the type of correlation method (Spearman or Pearson) for which the correlation matrix will be computed.

#### Widgets
- **`correlation_channels`**: Allows multiple selection of channels for which to calculate the correlation coefficients.
  - **Type**: SelectMultiple (Options are DataFrame columns)
- **`correlation_select`**: Allows selection of the correlation method to be used (Spearman or Pearson).
  - **Type**: RadioButtons

#### Output
Displays the widgets for channel and correlation method selection.

#### Additional Notes
The selected channels and correlation method will be used in the next cell to plot the correlation heatmap.

### Cell 12: Generate Correlation Heatmap

#### Functionality
Generates a heatmap to visualize the correlations between the selected channels. The heatmap provides insight into how the channels are related to each other.

#### Functions
- **`plot_correlation_heatmap(df, channels, method)`**: 
    - Plots a correlation heatmap for the given DataFrame, channels, and correlation method.
    
#### Variables
- **`transformed_data`**: Transforms the original data by applying the square root.
- **`scaled_data`**: Scales the transformed data using the previously selected scaler.

#### Output
Displays the correlation heatmap.

#### Additional Notes
The cell uses the square root transformation and scaling to prepare the data for correlation analysis.

### Cell 13: Save Correlation Heatmap to File (Optional)

#### Functionality
Contains code for saving the generated correlation heatmap to a PNG file.

#### Output
No output unless the code is uncommented and executed.

#### Additional Notes
To save the heatmap, uncomment the lines and execute the cell. The heatmap will be saved as a PNG file in the specified path.
