import cv2 
import numpy as np 
from scipy import signal

def Pixelate( image, pix_w, pix_h):
    """ Pixelate image 
    (ref) https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python
    Applying for 
                 * RGB 
                 * Gray 
    """    
    H, W = image.shape[:2]

    out_temp = cv2.resize(image, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR) # downsample 
    output = cv2.resize(out_temp, (W, H), interpolation=cv2.INTER_NEAREST) # upsample 

    return output 



def Phosephene(image, imgShape, pixShape, strength=7):
    faceSize = int(imgShape / pixShape)
    
    kernel = gkern(faceSize, std=strength)
    phosephene_kernel = np.zeros((imgShape,imgShape), dtype=float)


    for y in range(0, imgShape, faceSize ):
        for x in range(0, imgShape, faceSize):
            """
            (ref) https://github.com/ashushekar/image-convolution-from-scratch
            """
            phosephene_kernel[y:y+faceSize, x:x+faceSize] = kernel

    
    phosephene_img = (image * phosephene_kernel).astype('uint8')
    return phosephene_img





def gkern(kernlen=21, std=2):
    """Returns a 2D Gaussian kernel array.
    (ref) https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d