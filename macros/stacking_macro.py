# @File(label="First Image", style="file") imgPath1
# @File(label="Second Image", style="file") imgPath2
# @File(label="Output File", style="file") outPath
# @DatasetIOService io
# @CommandService command

from ij import IJ, ImagePlus, ImageStack

def run():
    # Open the two images
    img1 = IJ.openImage(str(imgPath1))
    img2 = IJ.openImage(str(imgPath2))

    # Create an empty stack
    stack = ImageStack(img1.getWidth(), img1.getHeight())

    # Add the first and second images to the stack
    stack.addSlice("Image 1", img1.getProcessor())
    stack.addSlice("Image 2", img2.getProcessor())

    # Create a new ImagePlus from the stack
    stackImage = ImagePlus("StackedImage", stack)

    # Save the stacked image
    IJ.save(stackImage, str(outPath))

run()
