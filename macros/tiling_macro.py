# @File(label="Input Image", style="file") imagePath
# @Integer(label="Number of Tiles", value=2) numTiles
# @DatasetIOService io
# @CommandService command

from ij import IJ, ImagePlus, ImageStack
from ij.plugin import Duplicator
from ij.gui import Roi

def run():
    # Open the image stack
    image = IJ.openImage(str(imagePath))

    # Calculate the width of each tile
    tileWidth = image.getWidth() // numTiles

    # Get the duplicator plugin to duplicate sections of the stack
    duplicator = Duplicator()

    # Loop over the tiles
    for i in range(numTiles):
        # Define the ROI for the current tile
        roi = Roi(i * tileWidth, 0, tileWidth, image.getHeight())
        image.setRoi(roi)

        # Duplicate the part of the stack for the current tile
        tileImage = duplicator.run(image)

        # Save the tile as a separate image
        IJ.save(tileImage, str(imagePath) + "_tile_" + str(i) + ".tiff")

run()
