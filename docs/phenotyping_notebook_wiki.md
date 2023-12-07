# Analysis Notebook

This wiki aims to briefly explain the functions and or requirements for each cell of the Analysis Notebook. Note that the cell numbers in this wiki must not be confused with the execution count (in brackets left of a cell) which indicates the cell's position in the execution order.

### Cell 1: Import Libraries and Set Up Environment

#### Functionality
This cell imports all the necessary Python libraries required for the notebook to function. These libraries support various operations like data manipulation, statistical analysis, image processing, and interactive widgets.

#### Libraries Imported
- **Standard Libraries**: `codecs`, `io`, `os` for basic Python operations.
- **Data Manipulation**: `pandas` (as `pd`) and `numpy` (as `np`) for data manipulation and mathematical operations.
- **Data Visualization**: `matplotlib.pyplot` (as `plt`), `plotly.express` (as `px`), and `seaborn` (as `sns`) for plotting and data visualization.
- **Image Processing**: `cv2` for image processing tasks.
- **Interactive Widgets**: `ipywidgets` (as `widgets`), `IPython.display` for interactive user interfaces.
- **Dimensionality Reduction and Clustering**: `umap`, `phenograph`, `sklearn.preprocessing` for machine learning tasks.
- **Statistical Analysis**: `scipy.stats` (as `stats`) for statistical functions.
- **Others**: `warnings` for suppressing warning messages, `RobustScaler`, `StandardScaler`, `MinMaxScaler` for data scaling, `numba.jit` for just-in-time compilation.

#### Special Commands
`%matplotlib widget`: This magic command enables the interactive Matplotlib backend for Jupyter notebooks.

#### Additional Notes
Warning messages are optionally suppressed using the `warnings` library to make the notebook output cleaner.

---

### Cell 2: Data Upload and Initialization

#### Functionality
Initializes widgets for data upload and sets up a function to read and format the uploaded CSV file.

#### Widgets
- **`input_file`**: Upload widget for single-cell CSV files.
    - **Type**: File (CSV)
- **`working_directory`**: Text box for the working directory path.
    - **Type**: Text

#### Function: `on_value_change(change)`
Updates global variables `singlecell_df` and `PATH` and briefly displays the uploaded DataFrame.

#### Output
Displays the file upload and directory path widgets, along with a temporary view of the uploaded DataFrame.

---

### Cells 3 & 4: Raw Histogram Plotting and Saving

#### Functionality
Generates a raw histogram plot for a selected channel from the uploaded single-cell data. Allows saving of the plot.

#### Widgets
- **`histogram_select`**: Radio buttons to select a channel for the histogram.
    - **Type**: Radio Buttons (Options are DataFrame columns)
    
#### Functions
- **`update_histogram(change, data, channel, cluster)`**: 
    - Updates and displays the histogram based on the selected channel.
    - Optional parameters `channel` and `cluster` for further filtering.
	- Cell 4 contains a line of code for saving the histogram plot. It is commented out by default.

#### Output
Displays radio buttons for channel selection and the histogram plot side by side.

#### Additional Notes
To save the histogram, uncomment the line in Cell 4 and execute it.

---

### Cells 5 & 6: Channel Normalization and DataFrame Display

#### Functionality
Cell 5 sets up widgets for entering pixel size and selecting channels for normalization. Cell 6 performs the normalization based on the selected options.

#### Widgets
- **`pixelsize_select`**: Text box for entering the pixel size in µm.
    - **Type**: Text
- **`normalise_channels`**: Multiple select box for channels to normalize.
    - **Type**: Multiple Select (Options are DataFrame columns)
- **`binary_select_norm`**: Radio buttons to decide whether to display the modified DataFrame.
    - **Type**: Radio Buttons ("Yes", "No")

#### Functions
Cell 6 contains code that normalizes selected columns by the area of cell objects, also adjusting for pixel size if entered. It uses a copy of the original DataFrame, `normalised_df`, to store the normalized data.

#### Output
Displays an interface to select pixel size, channels for normalization, and an option to display the modified DataFrame.

#### Additional Notes
- If no pixel size is entered for normalization, a warning message will appear, and the original data will be used.
- To display the modified DataFrame, select "Yes" on the `binary_select_norm` widget.

---

### Cells 7, 8 & 9: Outlier Filtering and Data Display

#### Functionality
Cells 7 and 8 contain functions for outlier removal based on percentiles or Z-scores. Widgets for selecting the filtering method and channels are also provided. Cell 9 includes a line for saving the filtered DataFrame.

#### Widgets
- **`filter_channels`**: Multiple select box to choose channels for outlier filtering.
    - **Type**: Multiple Select (Options are DataFrame columns)
- **`outlier_method`**: Radio buttons to select the outlier filtering method.
    - **Type**: Radio Buttons ("Percentiles", "Z-Score")
- **`binary_select_filter`**: Radio buttons to choose whether to display the modified DataFrame.
    - **Type**: Radio Buttons ("Yes", "No")

#### Functions
- **`remove_outliers_percentiles(df, columns)`**: Removes outliers based on percentile values.
- **`remove_outliers_zscore(df, columns, n_std)`**: Removes outliers based on Z-scores.

#### Output
Displays widgets for channel selection, outlier method, and an option to display the modified DataFrame.

#### Additional Notes
- Cell 8 checks if the data was previously normalized and uses the appropriate DataFrame (`normalised_df` or `original_df`) for filtering.
- Number of filtered cell events will be printed.
- To save the filtered DataFrame to a CSV file, uncomment the line in Cell 9 and execute it.

---

### Cells 10 & 11: Plotting Histograms for Modified Data

#### Functionality
Cell 10 sets up widgets for selecting a data version (normalized or filtered) and a channel for histogram plotting. It then plots the selected histogram. Cell 11 includes a line for saving the histogram.

#### Widgets
- **`histogram_select`**: Radio buttons for selecting a channel for the histogram.
    - **Type**: Radio Buttons (Options are DataFrame columns)
- **`hist_version_select`**: Radio buttons for choosing the version of data (normalized or filtered) to use for plotting.
    - **Type**: Radio Buttons (Options depend on available data)

#### Functions
- **`update_hist_version(change)`**: Updates the histogram based on selected data version and channel.

#### Output
Displays radio buttons for data version and channel selection, along with the histogram plot.

#### Additional Notes
- The cell checks which versions of the data are available (normalized, filtered) and adjusts the widget options accordingly.
- To save the histogram, uncomment the line in Cell 11 and execute it.

---

### Cell 12: Data Selection for Downstream Analysis

#### Functionality
Determines the version of data (filtered, normalized, or original) to be used for downstream analysis and informs the user.

#### Output
Prints a message indicating which version of the data will be used for downstream analysis.

#### Additional Notes
- If outlier filtering has been performed, `filtered_df` will be used.
- If only size normalization has been done, `normalised_df` will be used.
- If neither has been performed, the original `singlecell_df` will be used.

---

### Cell 13: Dynamic Heatmap Channel Visualization on Raw Image

#### Functionality
This cell creates a dynamic heatmap channel overlay on a raw image. The heatmap colors indicate the intensity of the selected channel. The user uploads a .png segmentation mask and a raw or processed sample image (.tiff format).

#### Widgets
- **`mask_upload`**: Upload widget for the segmentation mask.
    - **Type**: File (PNG)
- **`overlay_image_upload`**: Upload widget for the raw or processed image.
    - **Type**: File (TIFF)
- **`column_select`**: Dropdown menu for selecting the channel to visualize.
    - **Type**: Dropdown (Options are DataFrame columns)
- **`load_data_button`**: Button to load the uploaded data and visualize the heatmap.
    - **Type**: Button

#### Functions
- **`read_uploaded_file(upload_widget)`**: Reads uploaded file into a NumPy array.
- **`update_color_mask_numba(img, color_mask, cell_ids, cell_colors)`**: Updates the color mask based on selected channel intensities.
- **`plot_heatmap(...)`**: Plots the heatmap based on various parameters like transparency and channel intensity limits.
- **`load_data(btn)`**: Main function that initializes the heatmap plotting.

#### Sliders
- **`Limits`**: Slider to adjust the upper and lower thresholds for channel intensity.
- **`Transparency`**: Slider to adjust the transparency of the overlayed channel intensities.

#### Output
Displays the upload widgets, the dropdown for channel selection, and the 'Load data and visualize' button. Once the button is clicked, the heatmap overlay on the raw image will be displayed along with sliders to adjust transparency and intensity limits.

#### Additional Notes
- Move the 'Limits' slider to adjust upper and lower thresholds for channel intensity.
- Move the 'Transparency' slider to adjust the transparency of the overlayed cell intensities.

---

### Cells 14 & 15: Channel Selection and Data Scaling for Clustering

#### Functionality
- **Cell 14**: Provides a multiple select box to choose channels for clustering.
- **Cell 15**: Scales the selected data to prepare it for clustering. Two scaling methods are available: Robust scaling and MinMax scaling.

#### Widgets
- **`cluster_channels`**: Multiple select box for channels to include in clustering.
    - **Type**: Multiple Select (Options are DataFrame columns)

#### Functions
**Cell 15** contains code that scales the data using either Robust scaling or MinMax scaling. The scaling method to use can be selected by commenting/uncommenting the respective lines.

#### Output
- **Cell 14**: Displays the multiple select box for channel selection.
- **Cell 15**: Scales the data based on the selected method and channels.

#### Additional Notes
- Robust scaling is more resilient to outliers and is the default scaling method. To switch to MinMax scaling, uncomment the relevant lines in Cell 15.
- For more information on data scaling methods, refer to the [scikit-learn documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) and [here](https://scikit-learn.org/stable/modules/preprocessing.html).

---

### Cell 16: PhenoGraph Clustering

#### Functionality
Runs PhenoGraph clustering on the scaled data based on selected parameters. Provides widgets to input the clustering parameters.

#### Widgets
- **`k_text`**: Text box to input the k-value for PhenoGraph.
    - **Type**: Text (Integer, Placeholder: 30)
- **`resolution_parameter_text`**: Text box to input the resolution parameter for Leiden clustering.
    - **Type**: Text (Float, Placeholder: 1.0)
- **`seed_text`**: Text box to input the initial seed for randomization.
    - **Type**: Text (Integer or empty, Placeholder: 42)
- **`update_button`**: Button to execute the clustering algorithm.
    - **Type**: Button

#### Functions
**`run_clustering(button)`**: Callback function that runs the PhenoGraph clustering algorithm with the given parameters.

#### Output
Displays text boxes for inputting k-value, resolution parameter, and initial seed, along with a button to run the clustering algorithm.

#### Additional Notes
- For more information on PhenoGraph and its parameters, refer to the [PhenoGraph GitHub repository](https://github.com/dpeerlab/phenograph) and the corresponding publications: [Phenograph](https://doi.org/10.1016/j.cell.2015.05.047) and [Leiden algorithm](https://doi.org/10.1038/s41598-019-41695-z).
- Leaving the 'Initial seed' parameter unset will result in a random starting seed, leading to slightly different results each run.

---

### Cell 17: UMAP Dimensionality Reduction and Visualization

#### Functionality
Executes Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction on the scaled and clustered data. It then visualizes the UMAP embedding, coloring the points based on cluster assignments.

#### Functions
- Uses the UMAP algorithm from the `umap-learn` library to perform dimensionality reduction.
- The UMAP embedding is converted to a DataFrame and visualized using Plotly Express.

#### Output
Displays a scatter plot of the UMAP embedding, where each point represents a cell and the color indicates the cluster assignment.

#### Additional Notes
- Default settings for UMAP's hyperparameters are used. For custom settings, you may uncomment the cell below.
- For more information on UMAP and its parameters, refer to the [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/index.html) and the corresponding [publication](https://doi.org/10.48550/arXiv.1802.03426).
- Leaving the 'random initiation state' parameter unset will result in a random starting state, causing slightly different results each run.

---

### Cell 18: Generate Cluster Heatmap Data

#### Functionality
Prepares the data for generating a cluster heatmap by calculating z-scores for each channel's median (or mean) values for every cluster.

#### Functions
- Creates a list of DataFrames, each representing a unique cluster.
- Calculates the median (or mean, if uncommented) of each channel for every cluster.
- Applies z-score scaling to these median (or mean) values for better heatmap visualization.

#### Output
Creates a DataFrame `heatmap_clusters` containing the z-scored median (or mean) values for each channel and cluster.

#### Additional Notes
Uncomment the section for using mean values instead of median for heatmap calculation.
  
### Cell 19: Interactive Cluster Heatmap

#### Functionality
Displays an interactive widget to select channels and update the cluster heatmap. The heatmap shows how each cluster's median (or mean) values differ from the global median (or mean) for each selected channel.

#### Functions
- Uses Seaborn to plot the heatmap.
- Allows users to select channels to include in the heatmap and to specify a title.
  
#### Output
Displays an interactive widget that enables users to select channels, enter a title, and generate an updated heatmap.

#### Additional Notes
Click the "Update Heatmap" button to refresh the heatmap based on the selected channels.

### Cell 20: Save Cluster Heatmap

#### Functionality
Code for saving the cluster heatmap to a PNG file is provided.

#### Functions
Utilizes Matplotlib's `savefig` method to save the heatmap.
  
#### Additional Notes
The line for saving the heatmap is commented out by default. Uncomment it to execute the save operation.

---

### Cell 21: Prepare non-Normalised Data for Cluster and Channel-based Histograms

#### Functionality
Prepares a DataFrame named `export_df` to be used for plotting histograms based on selected clusters and channels. It reverts the size normalization if pixel size was initially set.

#### Functions
- Copies `filtered_df` into `export_df`.
- Divides the 'Area' and 'EquivalentDiameter' columns by the square and the first power of the pixel size, respectively, if pixel size was provided.

#### Additional Notes
This code is commented out by default. Uncomment it to execute.

### Cell 22: Interactive Cluster and Channel-based Histograms

#### Functionality
Displays an interactive widget that allows users to choose a specific cluster and channel to visualize in a histogram.

#### Functions
- Creates RadioButtons for selecting a cluster and a channel.
- Observes changes in the widgets and updates the histogram accordingly.

#### Output
Displays an interactive widget for cluster and channel selection. A histogram is generated based on the selected cluster and channel.

### Cell 23: Save Cluster and Channel-based Histogram

#### Functionality
Provides a code snippet to save the generated histogram to a PNG file.

#### Functions
Utilizes Matplotlib's `savefig` method to save the histogram.

#### Additional Notes
The line for saving the histogram is commented out by default. Uncomment it to execute the save operation.

---

### Cell 24: Select Clusters for CSV Export

#### Functionality
- Displays an interactive widget that allows users to select multiple clusters that will be saved as CSV files.

#### Functions
- Creates a `SelectMultiple` widget with options populated from the unique clusters in `export_df`.

#### Output
- Displays an interactive widget for cluster selection.

### Cell 25: Save Selected Clusters as CSV Files

#### Functionality
- Provides code snippets to save the data of selected clusters to individual CSV files. 

#### Functions
- Iterates through the selected clusters and filters `export_df` to only contain rows corresponding to each selected cluster.
- Utilizes Pandas' `to_csv` method to save each filtered DataFrame to a CSV file.

#### Additional Notes
- The lines for saving the data to CSV files are commented out by default. Uncomment them to execute the save operation.

---

### Cell 26: Dynamic Heatmap Channel Visualization on Raw Image (for uploaded cluster data)

#### Functionality
This cell creates a dynamic heatmap channel overlay on a raw image. The heatmap colors indicate the intensity of the selected channel. The user uploads a .png segmentation mask and a raw or processed sample image (.tiff format).

#### Widgets
- **`mask_upload`**: Upload widget for the segmentation mask.
    - **Type**: File (PNG)
- **`overlay_image_upload`**: Upload widget for the raw or processed image.
    - **Type**: File (TIFF)
- **`column_select`**: Dropdown menu for selecting the channel to visualize.
    - **Type**: Dropdown (Options are DataFrame columns)
- **`load_data_button`**: Button to load the uploaded data and visualize the heatmap.
    - **Type**: Button

#### Functions
- **`read_uploaded_file(upload_widget)`**: Reads uploaded file into a NumPy array.
- **`update_color_mask_numba(img, color_mask, cell_ids, cell_colors)`**: Updates the color mask based on selected channel intensities.
- **`plot_heatmap(...)`**: Plots the heatmap based on various parameters like transparency and channel intensity limits.
- **`load_data(btn)`**: Main function that initializes the heatmap plotting.

#### Sliders
- **`Limits`**: Slider to adjust the upper and lower thresholds for channel intensity.
- **`Transparency`**: Slider to adjust the transparency of the overlayed channel intensities.

#### Output
Displays the upload widgets, the dropdown for channel selection, and the 'Load data and visualize' button. Once the button is clicked, the heatmap overlay on the raw image will be displayed along with sliders to adjust transparency and intensity limits.

#### Additional Notes
- Move the 'Limits' slider to adjust upper and lower thresholds for channel intensity.
- Move the 'Transparency' slider to adjust the transparency of the overlayed cell intensities.

---