## Data Analysis

As MeXpose is composed of multiple existing analysis tools, as well as custom scripts, the individual analysis steps do require some knowledge of the software used. However, individual tools can be replaced with any other personal preference as long as the final input requirements [File Types - README] for the analysis python scripts are fulfilled.

Resources to become familiar with the used software packages can be found here:

- [Fiji/ImageJ](https://imagej.net/learn/)
- [Cellpose](https://cellpose.readthedocs.io/en/latest/index.html)
- [Cellprofiler](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.1/index.html)
- [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/notebook.html)

[The Scientific Community Image Forum](https://forum.image.sc) is also a great place to learn and look for help.

The following sections will cover the minimum required steps/setup for each of the used tools. The following instructions assume that the aliases included in the aliases.sh file are used. Adapt the commands accordingly for any changes or custom aliases.

### Fiji

Fiji is used to preprocess raw image data. It can be run either with a GUI for interactive use or in headless mode for a more streamlined workflow when using multiple datasets.

#### GUI

To run the Fiji-GUI run the command ```fiji-gui``` from the containers command line.  
Fiji is designed to be run as a graphical program and provides a multitude of functionalities.  
Within MeXpose, Fiji is used to preprocess raw images and prepare them for segmentation or downstream data extraction and analysis.

A typical Fiji workflow could look like this:

**Segmentation specific**

- Load the relevant channels for segmentation either as a stack or as seperate images  
```File --> Open```
- Stack the individual images (not needed if a stack is used as input)
```Image --> Stacks --> Images to Stack```
- Apply desired preprocessing steps, e.g. filtering, removing outliers, enhancing contrast  
```Process --> Filters --> Gaussian/Median```  
```Process --> Noise --> Remove Outliers```  
```Process --> Enahnce Contrast``` or ```Image --> Adjust --> Brightness/Contrast```
- Save the preprocessed segmentation stack as a TIFF stack  
```File --> Save As --> TIFF```

**Analysis specific**

Be are that any modifications to the raw image data will carry over to all downstream analysis steps including quantification.

- Load the relevant channels downstream analysis as a stack or as seperate images  
```File --> Open```
- Apply desired preprocessing steps, e.g. filtering, removing outliers  
```Process --> Filters --> Gaussian/Median```  
```Process --> Noise --> Remove Outliers```
- Save the preprocessed images as a tiff  
```File --> Save As --> TIFF```


#### Command Line

To run Fiji in headless mode without a GUI, MeXpose ships with custom aliases for four custom macros. The following macros are available and can be launched using the corresponding commands.

 - ```fiji-stack``` Takes two images and saves them as a combined stack. Use of TIFF files is recommended for input and required for output files.  
 **Usage**: ```fiji-stack /path/to/image1 /path/to/image2 /path/to/output.tiff```
 - ```fiji-tile``` Tiles the input image into the number of desired tiles 'n'.  
 **Usage**: ```fiji-tile /path/to/image number_of_tiles```
 - ```fiji-outliers``` Takes a directory of images and performs outlier filtering on a images contaiend within the directory. First applies a median filter with 'r=radius' and subsequently replaces all pixels devating from the calculated median by more than 't=threshold'.  
**Usage**: ```fiji-outliers /path/to/images 'r=radius' 't=threshold```
- ```fiji-filter``` Performs gaussian or median filtering with 'r=radius' on all images in a directory.  
**Usage**: ```fiji-filtering /path/to/images 'f=filter type' 'r=radius```


To run any custom macros other than the ones mentioned one should use the following command:  
```fiji --headless --run /path/to/your/macro```

### Cellpose

Cellpose is used to perform cell segmentation. It can be run either with a GUI for interactive use or in headless mode for a more streamlined workflow when using multiple datasets.

#### GUI

To run the Cellpose-GUI simply enter the command ```cellpose``` in the containers command line.  
The Cellpose-GUI serves three main functions:

- Running the segmentation algorithm.
- Manually labelling data.
- Fine-tuning a pretrained cellpose model on your own data.

A general workflow for performing segmentation using the Cellpose GUI could look as follows:

1. Load the desired image
2. Set 'chan to segment' - Set this to your cytoplasm/membrane channel if you want to  use a model trained on cell images with cytoplasm markers. Otherwise if you want to segment using only a nucleus signal, set this to the nucleus channel.
3. Set 'chan2' - Set this to your nucleus channel if using a cytoplasmn/membrane channel to segment. This is optional but recommended.
4. Set the cell diameter or use the auto calibrate function
5. Select a pretrained model for segmentation

For more information on how to use the Cellpose GUI for segmentation, to train your own models and the effects of the multiple settings, please consult the Cellpose documentation as well as the video below.

- [Cellpose GUI](https://cellpose.readthedocs.io/en/latest/gui.html#using-the-gui)
- [Cellpose settings](https://cellpose.readthedocs.io/en/latest/settings.html#settings)
- [Video - Cellpose2: human-in-the-loop model training](https://www.youtube.com/watch?v=3Y1VKcxjNy4)

#### Command Line

To run Cellpose as a command line program without a GUI (often preferable for multiple images) Cellpose needs several command line flags. Below is an example command with the minimum required parameters.

```
cellpose --dir path/of/images/to/segment/ \
		 		 --pretrained_model cyto \
		 		 --diameter 0. \
		 		 --chan 2 \
		 		 --chan2 3 \
		 		 --save_png
```

A brief explanation of each parameter (taken from [Cellpose settings documentation](https://cellpose.readthedocs.io/en/latest/settings.html#settings)):

- ```--dir path/of/images/to/segment/``` The absolute path to the directory containing all images to run segmentation on.
- ```--pretrained_model cyto``` Model to use for running or starting training.
- ```--diameter``` Cell diameter, if 0 will use the diameter of the training labels used in the model, or with built-in model will estimate diameter for each image
- ```--chan``` Channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: 0
- ```--chan2``` Nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: 0
- ```--save_png``` Save masks as png and outlines as text file

For a more detailed explanation of Cellpose settings and the available command line parameters and CLI usage, please consult the Cellpose documentation here [Cellpose settings](https://cellpose.readthedocs.io/en/latest/settings.html#settings) and here [Cellpose CLI](https://cellpose.readthedocs.io/en/latest/cli.html).

### Cellprofiler

Cellprofiler is used to extract and preprocess single cell data. It can be run either with a GUI for interactive use or in headless mode for a more streamlined workflow when using multiple datasets.

#### GUI

To run Cellprofiler with a GUI simply enter the command ```cellprofiler-gui``` in the containers command line. To utalise external plugins with Cellprofiler, the plugins directory has to be set after the initial launch of the GUI. To do this navigate to ```File --> Preferences --> CellProfiler plugins directory``` and enter the following path ```'/home/mxp/plugins'```. Make sure to click ```save``` afterwards.  
Cellprofiler is designed to be run as a graphical program and provides a multitude of functionalities.  
Within MeXpose, the Cellprofiler-GUI fulfils (but is not limited to) the following tasks:

- Relating segmented cells with pixel-based data 
- Extracting single cell feature information, e.g. intensities and area.
- Filtering objects/cells adjoining the image borders.

A general workflow for single cell feature extraction using the Cellprofiler-GUI could look as follows (parentheses indicate the corresponding modules):

1. Load the desired images and segmentation mask (Images)
2. Set the names for your images (NamesAndTypes)
3. Convert segmentation mask to objects (ConvertImageToObjects)
4. Filter objects adjoining the image borders (FilterObjects)
5. Measure the object morphologies and intensities (MeasureObjectSizeShape, MeasureObjectIntensityMultichannel)
6. Export the data to a CSV (ExportToSpreadsheet)

For more information on how to use the Cellprofiler-GUI please consult the Cellprofiler manual, wiki and official tutorials.

- [Cellprofiler wiki](https://github.com/CellProfiler/CellProfiler/wiki#cellprofiler-wiki)
- [Cellprofiler manual (v4.2.5)](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.5/index.html)
- [Cellprofiler tutorials](https://tutorials.cellprofiler.org)

#### Command Line

To run Cellprofiler in headless mode without a GUI, Cellprofiler needs to be run with several command line flags. Below is an example command with the minimum required parameters.

```
cellprofiler -p /path/to/pipeline/file.cppipe \
			 -o /path/where/the/output/goes \
			 -i /path/with/input/files
```

**Note**: For convenience sake the parameters ```-c``` and ```-r``` which are required to run Cellprofiler in headless mode are already included in the bash alias in MeXpose. If you are using your own aliases, make sure to add these parameters.

A brief explanation of each parameter (taken from [Cellpose settings documentation](https://cellpose.readthedocs.io/en/latest/settings.html#settings)):

- ```-p /path/to/pipeline/file.cppipe``` The absolute path to the Cellprofiler pipeline file used to process the data.
- ```-o /path/where/the/output/goes``` The absolute path to the desired output directory.
- ```-i /path/with/input/files``` The absolute path of the input directory.

For a more information on Cellprofilers command line parameters and CLI usage, please consult the Cellprofiler wiki here [Getting started using CellProfiler from the command line](https://github.com/CellProfiler/CellProfiler/wiki/Getting-started-using-CellProfiler-from-the-command-line) and here [Adapting CellProfiler to a LIMS environment](https://github.com/CellProfiler/CellProfiler/wiki/Adapting-CellProfiler-to-a-LIMS-environment).