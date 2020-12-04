import cv2 
import numpy as np 
from scipy import signal
import skimage.measure 

def Pixelate( image, pix_w, pix_h):
    """ Pixelate image 
    (ref) https://stackoverflow.com/questions/55508615/how-to-pixelate-image-using-opencv-in-python
    (ref) https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy

    """    
    H, W = image.shape[:2]
    scale = int(H/pix_h)
    
#    out_temp = cv2.resize(image, (pix_w, pix_h), interpolation=cv2.INTER_CUBIC) # downsample 

    out_temp = skimage.measure.block_reduce(image, (scale, scale), np.max)  # MaxPooling 

    
    output = cv2.resize(out_temp, (W, H), interpolation=cv2.INTER_NEAREST ) # upsample 

    return output 



def Phosephene(image, imgShape, pixShape, strength=1):
    """
    Origi -> pixeltated -> Phosephene  
    """

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


def Phosephene32(img, imgShape ,pixShape, strength=1):
    """
    Origi -> maxPool_32 ->  Phosephene -> Upsampling 480x480
    """

    H, W = img.shape 
    faceSize = int(imgShape / pixShape)
    
    phosephene_face = np.zeros((imgShape,imgShape), dtype=float)  # zero_templete 
    kernel = gkern(faceSize, std=strength)
    
    for j in range(H):
        for i in range(W): 
            temp = np.ones((faceSize,faceSize)) * img[j,i]
            
            phosephene_temp =(temp * kernel).astype('uint8')
            phosephene_face[j*faceSize:j*faceSize+faceSize, i*faceSize:i*faceSize+faceSize] = phosephene_temp

    
    return  phosephene_face




def gkern(kernlen=21, std=2):
    """Returns a 2D Gaussian kernel array.
    (ref) https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d