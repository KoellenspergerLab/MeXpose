# Fiji macro launch scipts

# Launches the stacking macro with 3 Arguments;
# Arguments are paths for both input channels and desired output path for image stack
fiji-stack() {
    if [ "$#" -ne 3 ]; then
        echo "Usage: fiji-stack /path/to/image1 /path/to/image2 /path/to/output"
        return 1
    fi
    home/Fiji.app/ImageJ-linux64 --headless --run /home/MeXpose/macros/stacking_macro.py "imgPath1='$1', imgPath2='$2', outPath='$3'"
}


# Launches the tiling macro with 2 Arguments;
# Arguments are the image path and "n='the number of desired image tiles'"
fiji-tile() {
    if [ "$#" -ne 2 ]; then
        echo "Usage: fiji-tile /path/to/image number_of_tiles"
        return 1
    fi
    home/Fiji.app/ImageJ-linux64 --headless --run /home/MeXpose/macros/tiling_macro.py "imagePath='$1', numTiles=$2"
}


# Launches the outlier filtering macro with 3 Arguments;
# Arguments are the image directory path, "t=' how much greater the pixel's value must be compared to the median to be considered a hot pixel'"
# and "n='the size of the neighborhood around each pixel'"
fiji-outliers() {
    if [ "$#" -ne 3 ]; then
        echo "Usage: fiji-outliers /path/to/images 'n=neighborhoodSize' 't=thresholdFactor'"
        return 1
    fi
    home/Fiji.app/ImageJ-linux64 --headless --run /home/MeXpose/macros/outlier_macro.py "inputDir='$1', thresholdFactor=$2, neighborhoodSize=$3"
}


# Launches the gaussian/median filtering macro with 3 Arguments;
# Arguments are the image directory path, "f='the desired filtering method (Gaussian/Median)'"
# and "r='the desired pixel radius'"
fiji-filtering() {
    if [ "$#" -ne 3 ]; then
        echo "Usage: fiji-filtering /path/to/images 'f=filter type' 'r=radius'"
        return 1
    fi
    home/Fiji.app/ImageJ-linux64 --headless --run /home/MeXpose/macros/filtering_macro.py "inputDir='$1', filterType='${2#*=}', radius=${3#*=}"
}
