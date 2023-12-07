# @File(label="Input Directory", style="directory") inputDir
# @Integer(label="Radius", value=1) radius
# @Float(label="Threshold", value=50.0) threshold
# @DatasetIOService io
# @CommandService command

from ij import IJ, ImagePlus
from ij.process import ImageProcessor
from ij.plugin.filter import RankFilters
from java.io import File
from java.lang import System

def run():
    # Get all files in the directory
    directory = File(str(inputDir))
    files = directory.listFiles()

    # Sort the files alphabetically
    files = sorted(files, key=lambda x: x.getName())

    # Create a rank filter for median calculation
    rankFilter = RankFilters()

    # Loop over all files and apply the outlier removal
    for i, file in enumerate(files):
        image = IJ.openImage(str(file))
        ip = image.getProcessor().duplicate()
        
        # Apply a median filter
        rankFilter.rank(ip, radius, RankFilters.MEDIAN)
        
        # Compare original image and median-filtered image
        for y in range(image.height):
            for x in range(image.width):
                # If a pixel deviates from the median by more than the threshold, replace it with the median
                if abs(image.getProcessor().getPixel(x, y) - ip.getPixel(x, y)) > threshold:
                    image.getProcessor().putPixel(x, y, ip.getPixel(x, y))
        
        # Save the processed image
        IJ.save(image, str(file) + "_outliers_removed.tiff")
    
    # Force exit
    System.exit(0)

run()
