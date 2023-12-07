# @File(label="Input Directory", style="directory") inputDir
# @String(label="Filter Type", choices={"Gaussian", "Median"}) filterType
# @Float(label="Radius", value=1.0) radius
# @DatasetIOService io
# @CommandService command

from ij import IJ, ImagePlus
from ij.plugin.filter import RankFilters, GaussianBlur
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
    gaussianFilter = GaussianBlur()

    # Loop over all files and apply the filtering
    for i, file in enumerate(files):
        image = IJ.openImage(str(file))
        
        # Apply the filter
        if filterType == "Median":
            rankFilter.rank(image.getProcessor(), radius, RankFilters.MEDIAN)
        else:  # Gaussian
            gaussianFilter.blurGaussian(image.getProcessor(), radius)
        
        # Save the processed image
        IJ.save(image, str(file) + "_filtered.tiff")

    # Force exit
    System.exit(0)

run()
