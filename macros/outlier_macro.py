# @File(label="Input Directory", style="directory") inputDir
# @Integer(label="Neighborhood Size", value=3) neighborhoodSize
# @Float(label="Threshold Factor", value=2) thresholdFactor
# @DatasetIOService io
# @CommandService command

from ij import IJ, ImagePlus
from ij.process import ImageProcessor
from java.io import File
from java.lang import System

def run():
    directory = File(str(inputDir))
    files = directory.listFiles()

    files = sorted(files, key=lambda x: x.getName())

    for file in files:
        image = IJ.openImage(str(file))
        ip = image.getProcessor()
        
        # Process each pixel
        for y in range(1, image.height - 1):
            for x in range(1, image.width - 1):
                processPixel(ip, x, y, neighborhoodSize, thresholdFactor)

        # Save the processed image
        IJ.save(image, str(file) + "_hotpixels_removed.tiff")

def processPixel(ip, x, y, size, thresholdFactor):
    # Get the pixel value
    pixelValue = ip.getPixel(x, y)
    
    # Calculate the median of the surrounding pixels
    surroundingPixels = getSurroundingPixels(ip, x, y, size)
    medianValue = sorted(surroundingPixels)[len(surroundingPixels) // 2]
    
    # Check if the pixel value deviates significantly from the median
    if pixelValue > medianValue * thresholdFactor:
        # Replace hot pixel with the median value
        ip.putPixel(x, y, medianValue)

def getSurroundingPixels(ip, x, y, size):
    # Collects the pixels surrounding the specified pixel
    surroundingPixels = []
    offset = size // 2
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            if i == 0 and j == 0:
                continue
            surroundingPixels.append(ip.getPixel(x + i, y + j))
    return surroundingPixels

run()
System.exit(0)
