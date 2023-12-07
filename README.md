
<img src="https://raw.githubusercontent.com/KoellenspergerLab/MeXpose/master/graphical_workflow.jpg" width=65% height=65%>

---

## Welcome

MeXpose is a workflow designed to perform multiplexed image analysis, specialising in (but not limited to) analysis of metal distribution in LA-ICP-TOFMS data.

MeXpose comes in two variations:


- **Interactive**	- Based on software GUIs and interactive Jupyter notebooks; Aimed at optimising parameters for new sample types
- **Automated**	- Based on software run in headless mode and python scripts; Aimed at providing fast routine analysis for established samples

### Overview

The following table elucidates the basic functionality of both MeXpose variations, highlighting the key advantages/differences. All functions are provided either through implementation of existing software or through custom macros and scripts.

|             | Pre-processing (Fiji)                                                                            | Segmentation (Cellpose)                                         | Object measurements (Cellprofiler)                                     | Data analysis & visualisation (Jupyter/Python)                    |
|:------------|:-------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:-----------------------------------------------------------------------|:------------------------------------------------------------------|
| **Interactive** | Full Fiji functionality                                                                          | Iterative improvements through human-in-the-loop workflow       | Full Cellprofiler functionality; Allows for marker-panel configuration | Notebook based analysis; Enables parameter tuning for new samples |
| **Automated**   | Headless execution; Macros for stacking, tiling, hot pixel removal and gaussian/median filtering | Headless execution; Streamlined segmentation of multiple images | Headless execution; Fast data extraction with configured marker panels | Python script; Allows batch throughput of image sets              |

MeXpose is modular and and supports drop-in replacements for any used third-party software. To guarantee compatability with downstream data processing, respect the [file type](#file-types) requirements. 

---

## Installation

MeXpose is deployed as a docker container but can also be run locally by individually installing the required software and either cloning the mexpose github repository or downloading the scripts, notebooks and macros individually.

### Dependencies

- Containerised:
	+ Docker

- Local:
	+ Fiji
	+ Cellpose
	+ Cellprofiler
	+ Anaconda or other Python distribution
	+ MeXpose github repository

- Building container:
	+ Docker
	+ Dockerfile (MeXpose github)
	+ mexpose.yaml conda environment (MeXpose github)

### Docker

Install Docker Desktop for your respective platform [Linux](https://docs.docker.com/desktop/install/linux-install/)/[Mac](https://docs.docker.com/desktop/install/mac-install/)/[Windows](https://docs.docker.com/desktop/install/windows-install/). For Windows, make sure to use the WSL2 engine for better performance.

Adjust resources allocated to Docker Desktop [Linux](https://docs.docker.com/desktop/settings/linux/)/[Mac](https://docs.docker.com/desktop/settings/mac/)/[Windows](https://learn.microsoft.com/en-us/windows/wsl/wsl-config). We recommend 8GB of RAM and 4 performance CPU cores as a baseline.

#### Interactive Setup

X forwarding has been tested on both Linux and Windows and should work 'out of the box' when using the sample commands under [Running MeXpose](running-mexpose). Interactive use could not be tested on Mac OS. However, X forwading can be achieved on Mac OS using [XQuartz](https://www.xquartz.org/index.html).

**Note**: The MeXpose container currently does not support GPU integration. 

---

## Running MeXpose

#### Linux

To pull the MeXpose Docker image and run a container enter the following command in your terminal. Adjust the volume bind mount `--volume="/path/to/data/:/root/data/"` according to your directory structure.

```
docker run -it \
  --env="DISPLAY" \
  --env=LANG=C.UTF-8 \
  --env=LC_ALL=C.UTF-8 \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
  --volume="$HOME/.Xauthority:/root/.Xauthority" \
  --volume="/path/to/data/:/root/data/" \
  --net=host \
  --name=mexpose \
  koellenspergerlab/mexpose:0.1.3
```

The first environment flag as well as the first two volume bind mounts are required for X forwarding, passing graphical output from the container to the host system. The other two environment flags are used for setting the locale. To keep image size as small as possible, only POSIX, C and C.UTF-8 locales are available. If you require a different locale, either 1.) Set the locale in the Dockerfile and rebuild the MeXpose image or 2.) Install the *locales* package with the command below and [set the locale inside the container](https://help.ubuntu.com/community/Locale#Changing_settings_temporarily).

```apt-get update -y && apt-get install locales -y```

**Note:** If you have changed the locale settings in either of the above ways, remember to remove both `--env=LANG=C.UTF-8` and `--env=LC_ALL=C.UTF-8` from the launch command.

#### Windows

When running on Windows one has the option to run the container from a PowerShell or from within the WSL distribution. Both options will utilise the WSL engine, therefore performance is expected to be the same.

When running from within the WSL distribution, use the command provided for Linux systems above. For execution from a PowerShell use the following command. Adjust the volume bind mounts according to your directory structure.

```
docker run -it --rm `
  -e DISPLAY=:0 `
  -e LANG=C.UTF-8 `
  -e LC_ALL=C.UTF-8 `
  -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix `
  -v /run/desktop/mnt/host/wslg:/mnt/wslg `
  -v "\path\to\data:/root/data" `
  --name=mexpose `
  koellensperger-lab/mexpose:latest
```

The first environment flag as well as the first two volume bind mounts are required for X forwarding, passing graphical output from the container to the host system. The other two environment flags are used for setting the locale. To keep image size as small as possible, only POSIX, C and C.UTF-8 locales are available. If you require a different locale, either 1.) Set the locale in the Dockerfile and rebuild the MeXpose image or 2.) Install the *locales* package with the command below and [set the locale inside the container](https://help.ubuntu.com/community/Locale#Changing_settings_temporarily).

```apt-get update -y && apt-get install locales -y```

**Note:** If you have changed the locale settings in either of the above ways, remember to remove both `-e LANG=C.UTF-8` and `-e LC_ALL=C.UTF-8` from the launch command.

### File Types

MeXposes expects 16bit TIFF files as raw image data input (32bit images will work, albeit with limitations in Fiji preprocessing), PNG files for the segmentation masks and CSV files for single-cell and quantification standards data. Due to the use of python for all data analysis steps, support for other file formats can be easily integrated by changing the respective code sections for data loading.

### Usage

A typical MeXpose interactive workflow would look like this:

1. **Preprocessing in Fiji:** stacking, smoothing/filtering and tiling of segmentation relevant channels; Outlier filtering and smoothing/filtering of relevant channels.
2. **Cell Segmentation using Cellpose:** After initial model selection, iterative model performance improvements through several rounds of manual annotations and retraining.
3. **Feature extraction and data export in Cellprofiler:** Setting up a marker panel-specific Cellprofiler project; Filtering out cells touching the image borders; Selecting and exporting features of interest (intensities, morphology)
4. **Exploratory data analysis using Jupyter Notebooks:** Initial inspection of the data through histograms; Phenotyping of cells using clustering & dimensionality reduction, cluster heatmaps and channel intensity heatmap overlays; Export of PNGs and CSV data of relevant cell phenotype data.
5. **Quantification using Jupyter Notebooks:** Quantification of isotopes on a single cell level.

---

## Citing MeXpose

Please cite the following paper when using MeXpose in your work:

---

## Licensing

The code included with MeXpose, namely all python scripts, jupyter notebooks and fiji macros, is distributed under the MIT license. The MeXpose container hosted on DockerHub is distributed under the GPLv3 license.
