// Variables
def outDir = "/mnt/ssd/Data/3DTumorModell/Slides/slices_png"

i = 1

def project = getProject()
for (entry in project.getImageList()) {

    // Default Objects
    def imageData = entry.readImageData()
    def server = imageData.getServer()
    
    // Extract slidename
    def imageName = entry.getImageName().split('\\.')[0] + ".png"
    
    // Define image name
    // def imageName = "image_" + String.format("%03d", i) + ".jpeg"
    
    // Write the full image downsampled by a factor of 4
    def requestFull = RegionRequest.createInstance(server, 4)
    writeImageRegion(server, requestFull, buildFilePath(outDir, imageName))

    i = i + 1
}