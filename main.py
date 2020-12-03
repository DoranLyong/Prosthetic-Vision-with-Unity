""" 
Code author : DoranLyong 

Reference : 
* https://docs.python.org/3.7/library/socketserver.html
* https://webnautes.tistory.com/1382
* https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples
* http://blog.cogitomethods.com/visual-analytics-using-opencv-and-realsense-camera/
"""
import socketserver
import socket
from queue import Queue
from _thread import *

import cv2 
import numpy as np 
from scipy import signal
import pyrealsense2 as d435
import skimage.measure 


from lib.TCPHandler import MyTCPHandler
from lib.tools import Pixelate, Phosephene, Phosephene32
from model.HED import CropLayer
from cfg import exp_cfg


# _Set queue 
enclosure_queue = Queue() 


# _Configures of depth and color streams 
pipeline = d435.pipeline()
config = d435.config()
config.enable_stream(d435.stream.depth, 640, 480, d435.format.z16, 30)
config.enable_stream(d435.stream.color, 640, 480, d435.format.bgr8, 30)


pix_h, pix_w = exp_cfg["pixSize"]
H, W = exp_cfg["imgShape"]
strength = exp_cfg["Strength"]

#! [Register]
cv2.dnn_registerLayer('Crop', CropLayer)
#! [Register]

# Load the model 
net = cv2.dnn.readNet(cv2.samples.findFile(exp_cfg["prototxt"]), cv2.samples.findFile(exp_cfg["hed_pretrained"]))



# _ D435 process 
def D435(queue):
    
    
    print("D435 processing", end="\n ")
    pipeline.start(config) # _Start streaming

    try:
        while True: 
            # _Wait for a coherent pair of frames: depth and color 
            frames = pipeline.wait_for_frames()            
            depth_frame, color_frame = (frames.get_depth_frame(), frames.get_color_frame())

            if not (depth_frame and color_frame): 
                print("Missing frame...", end="\n")
                continue

            # _Convert <pyrealsense2 frame> to <ndarray>
            depth_image = np.asanyarray(depth_frame.get_data()) # convert any array to <ndarray>
            color_image = np.asanyarray(color_frame.get_data())
            
            # _ Crop (H,W)=(480, 640) ro (480, 480)
            img_H, img_W = color_image.shape[:2]
            rgb_480 = color_image[:,80: img_W-80 ,: ] # (width-480)/2 = 80 
            depth_480 = depth_image[:,80: img_W-80 ] # (width-480)/2 = 80 

            cv2.imshow("rgb_480 origin", rgb_480)
            cv2.imshow("depth_480 origin", depth_480)


            # _Apply colormap on depth image 
            #  (image must be converted to 8-bit per pixel first)            
            
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_480, alpha=0.05), cv2.COLORMAP_BONE)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
            #depth_colormap  = cv2.bitwise_not(depth_colormap ) # reverse image

            #print("Depth map shape = ", depth_colormap.shape)   



            """ Color convert
            """
            gray = cv2.cvtColor(rgb_480 , cv2.COLOR_BGR2GRAY)
            depth = cv2.cvtColor(depth_colormap,  cv2.COLOR_BGR2GRAY)


            """Canny
            """
            canny = cv2.Canny(gray,50, 255)
            cv2.imwrite('./canny.bmp',  canny)


            gray_depth_canny = np.hstack((gray, depth, canny))
            
            cv2.namedWindow("gray + depth_gray + canny", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("gray + depth_gray + canny", gray_depth_canny)


            """HED
            """
#            inp = cv2.dnn.blobFromImage(rgb , scalefactor=1.0, size=(W, H),
#                               mean=(104.00698793, 116.66876762, 122.67891434),
#                               swapRB=False, crop=False)
#
#            net.setInput(inp)                               
#            out = net.forward() 
#            out = out[0, 0]
#            cv2.imshow("HED", out)


            """ Pixelate image 
            """           
            pixSize = (pix_h, pix_w)
            pixelated_gray = Pixelate(gray , *pixSize)
            pixelated_depth = Pixelate(depth , *pixSize)
            pixelated_canny = Pixelate(canny, *pixSize)
#            pixelated_HED = Pixelate(out , *pixSize)

                      
#            ret2, pixelated_HED = cv2.threshold(pixelated_HED*255,200,255, cv2.THRESH_BINARY)
#            cv2.imshow("pixelated_HED", pixelated_HED )     
            
            cv2.imwrite('./pix_canny.bmp', pixelated_canny)
            
            pixelated_gray_depth_canny = np.hstack((pixelated_gray, pixelated_depth, pixelated_canny ))
            cv2.imshow("pixelated_gray_depth_canny", pixelated_gray_depth_canny)       



            """ Phosephene image 
            """
            phosephene_gray = Phosephene(pixelated_gray, H, pix_h, strength=strength )
            phosephene_depth = Phosephene(pixelated_depth, H, pix_h, strength=strength)
            phosephene_canny = Phosephene(pixelated_canny, H, pix_h, strength=strength)
##            phosephene_HED = Phosephene(pixelated_HED*255  , H, pix_h, strength=strength)

            phosephenes = np.hstack((phosephene_gray, phosephene_depth, phosephene_canny ))                       
            cv2.imshow("phosephens_from_480", phosephenes)
#
##            cv2.imshow("phosephene_HED", phosephene_HED )
#
# 

            """ MaxPooling version 
            """
            scale = int(img_H/pix_h)  # 480/32 = 15. That is, you will get the 32x32 image after MaxPooling

            gray_32 = skimage.measure.block_reduce(gray, (scale, scale), np.max) # MaxPooling  (480,480) -> (32,32)
            depth_32 = skimage.measure.block_reduce(depth, (scale, scale), np.max) # MaxPooling  (480,480) -> (32,32)
            canny_32 = cv2.Canny(gray_32 ,50, 255)

            MaxPool_32 = np.hstack((gray_32, depth_32, canny_32))
            cv2.namedWindow("MaxPool_32", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("MaxPool_32", MaxPool_32)

#           
            phosephene_gray_32 = Phosephene32(gray_32, H, pix_h, strength=strength )
            phosephene_depth_32 = Phosephene32(depth_32, H, pix_h, strength=strength )
            phosephene_canny_32 = Phosephene32(canny_32, H, pix_h, strength=strength )

            phosephenes_32 = np.hstack((phosephene_gray_32, phosephene_depth_32, phosephene_canny_32))
            cv2.imshow("phosephenes_32", phosephenes_32)
#
#

#           
#            # _Encoding 
#            target_frame = phosephene_depth
#            
#            #print(target_frame.shape)
#            
#
#            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),80]  # 0 ~ 100 quality 
#            #encode_param = [cv2.IMWRITE_PNG_COMPRESSION,0] # 0 ~ 9 Compressiong rate 
#            #encode_param = [int(cv2.IMWRITE_WEBP_QUALITY),95]  # 0 ~ 100 quality 
#
#
#            result, imgencode = cv2.imencode('.jpg', target_frame, encode_param)  # Encode numpy into '.jpg'
#            data = np.array(imgencode)
#
#            stringData = data.tostring()   # Convert numpy to string
#            #print("byte Length: ", len(stringData))
#            queue.put(stringData)          # Put the encode in the queue stack
#
#

            cv2.waitKey(1)        

        
    finally: 
        cv2.destroyAllWindows()

        # _Stop streaming 
        pipeline.stop()



if __name__ == "__main__":

    # _Webcam process is loaded onto subthread
    start_new_thread(D435, (enclosure_queue,))  
    
    # _Server on
    HOST, PORT = socket.gethostname(), 8080 
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:    
        
        print("****** Server started ****** ", end="\n \n")     
        
        try: 
            server.serve_forever()
        
        except KeyboardInterrupt as e:
            print("******  Server closed ****** ", end="\n \n" )  