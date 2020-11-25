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

from lib.TCPHandler import MyTCPHandler
from lib.tools import Pixelate, Phosephene
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


            # _Apply colormap on depth image 
            #  (image must be converted to 8-bit per pixel first)            
            
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_BONE)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
            #depth_colormap  = cv2.bitwise_not(depth_colormap ) # reverse image

            #print("Depth map shape = ", depth_colormap.shape)   



            """ Resize images 
            """

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            depth = cv2.cvtColor(depth_colormap,  cv2.COLOR_BGR2GRAY)
            
            gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST )
            depth = cv2.resize(depth , (W, H), interpolation=cv2.INTER_NEAREST )


            cv2.imshow("gray", gray)
            cv2.imshow("depth", depth)

            

            """ Pixelate image 
            """           
            pixSize = (pix_h, pix_w )
            pixelated_gray = Pixelate(gray , *pixSize)
            pixelated_depth = Pixelate(depth , *pixSize)
            
          
            

            """ Phosephene image 
            """
            phosephene_gray = Phosephene(pixelated_gray, H, pix_h, strength=strength )
            phosephene_depth = Phosephene(pixelated_depth, H, pix_h, strength=strength)
            
        


            # _Encoding 
            target_frame = phosephene_depth
            
            #print(target_frame.shape)
            

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),80]  # 0 ~ 100 quality 
            #encode_param = [cv2.IMWRITE_PNG_COMPRESSION,0] # 0 ~ 9 Compressiong rate 
            #encode_param = [int(cv2.IMWRITE_WEBP_QUALITY),95]  # 0 ~ 100 quality 


            result, imgencode = cv2.imencode('.jpg', target_frame, encode_param)  # Encode numpy into '.jpg'
            data = np.array(imgencode)

            stringData = data.tostring()   # Convert numpy to string
            #print("byte Length: ", len(stringData))
            queue.put(stringData)          # Put the encode in the queue stack


            # __ Image show             
            images1 = np.hstack((pixelated_gray, pixelated_depth)) # stack both images horizontally            
            images2 = np.hstack((phosephene_gray , phosephene_depth)) 
            images = np.vstack((images1, images2))
            

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
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