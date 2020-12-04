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
import os 
import os.path as osp 
import sys 
ws_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(ws_dir, 'lib' ))
sys.path.insert(0, osp.join(ws_dir, 'model' ))
os.chdir(ws_dir) # change cwd


import cv2 
import numpy as np 
from scipy import signal
import pyrealsense2 as d435
import skimage.measure 


from lib.TCPHandler import MyTCPHandler
from lib.tools import Pixelate, Phosephene, Phosephene32
from model.HED import CropLayer
from cfg import exp_cfg
from args import argument_parser, SSD_classes
from lib.NMS import non_max_suppression_fast as NMS 


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


# Load the model for HED 
net = cv2.dnn.readNet(cv2.samples.findFile(exp_cfg["prototxt"]), cv2.samples.findFile(exp_cfg["hed_pretrained"]))

# Load the model for SSD 
ap = argument_parser()
args = vars(ap.parse_args())
SSD_net = cv2.dnn.readNetFromTensorflow(args["pb"], args["pbtxt"])
CLASSES = SSD_classes()


totalFrames = 0 

key_press = ord('q')


# _ D435 process 
def D435(queue):
    global key_press
    
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

#            cv2.imshow("rgb_480 origin", rgb_480)
#            cv2.imshow("depth_480 origin", depth_480)


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

            """Mobile SSD_v3 
            """
            bboxes = [] 
            if totalFrames % args["skip_frames"] == 0:
                
                SSD_net.setInput(cv2.dnn.blobFromImage(rgb_480, 0.007843, (480,480), 127.5 ))
                detections = SSD_net.forward()

                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > args["confidence"]:
                        idx = int(detections[0, 0, i, 1])

                        if CLASSES[idx] not in ["cup", "knife", "bottle", "backpack", "keyboard", 
                                                "book", "cell_phone", "person", "laptop" ] :
                            continue


                        box = detections[0, 0, i, 3:7] * np.array([480, 480 ,480, 480]) # (W, H) = (480,480)
                        (x1, y1, x2, y2) = box.astype("int")
                        bboxes.append((x1, y1, x2, y2))

            # ---- NMS algorithm ---- # 
            bboxes = np.array(bboxes)
            pick = NMS(bboxes, overlapThresh=0.3)

            mask_temp = np.zeros_like(gray).astype('uint8')
            masks = [] 
            for (x_start, y_start, x_end, y_end) in pick: 
                cv2.rectangle(rgb_480, (x_start, y_start), (x_end, y_end), (0,255,255), thickness=2, lineType=cv2.LINE_8)
                mask_temp[y_start:y_end, x_start:x_end] = 1 


            cv2.imshow("SSD_mobile", rgb_480)
#            gray = gray * mask_temp
#            depth = depth * mask_temp
#            cv2.imshow("mask", gray)




            """Edge - Canny
            """
            canny = cv2.Canny(gray, 80, 200)
#            canny = cv2.Laplacian(gray, cv2.CV_8U,ksize=5 )

#            cv2.imwrite('./canny.bmp',  canny)


            gray_depth_canny = np.hstack((gray, depth, canny))
            
            cv2.namedWindow("gray + depth_gray + edge", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("gray + depth_gray + edge", gray_depth_canny)


            """Edge - HED
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
            pixelated_canny = Pixelate(canny , *pixSize)
#            pixelated_HED = Pixelate(out , *pixSize)

                      
#            ret2, pixelated_HED = cv2.threshold(pixelated_HED*255,200,255, cv2.THRESH_BINARY)
#            cv2.imshow("pixelated_HED", pixelated_HED )     
            
#            cv2.imwrite('./pix_edge.bmp', pixelated_canny)
            
            pixelated_gray_depth_canny = np.hstack((pixelated_gray, pixelated_depth, pixelated_canny ))
            cv2.imshow("pixelated_gray_depth_edge", pixelated_gray_depth_canny)       



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
#            canny_32= cv2.Laplacian(gray_32, cv2.CV_8U,ksize=5 )
            canny_32 = cv2.Canny(gray_32 ,80, 255)

            MaxPool_32 = np.hstack((gray_32, depth_32, canny_32))
            cv2.namedWindow("MaxPool_32", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("MaxPool_32", MaxPool_32)

#           
            phosephene_gray_32 = Phosephene32(gray_32, H, pix_h, strength=strength )
            phosephene_depth_32 = Phosephene32(depth_32, H, pix_h, strength=strength )
            phosephene_canny_32 = Phosephene32(canny_32, H, pix_h, strength=strength )

            phosephenes_32 = np.hstack((phosephene_gray_32, phosephene_depth_32, phosephene_canny_32))
            cv2.imshow("phosephenes_32", phosephenes_32)




            key = cv2.waitKey(1)        


            # _Encoding 

            if key == 27:  # 'ESC'
                break
            
            elif key == -1: # no key_event 
                pass 

            else: 
                key_press = key 


            if False: 
                cast = mask_temp
            else: 
                cast = np.ones_like(mask_temp).astype('uint8')
             

            target = {ord('q'): gray*cast , ord('w'):pixelated_gray*cast   ,ord('e'): phosephene_gray*cast , ord('r'): phosephene_gray_32*cast , ord('1'): cv2.cvtColor(rgb_480 , cv2.COLOR_BGR2GRAY),
                      ord('a'): depth*cast , ord('s'): pixelated_depth*cast  , ord('d'): phosephene_depth*cast , ord('f'): phosephene_depth_32*cast , 
                      ord('z'): canny*cast , ord('x'): pixelated_canny*cast , ord('c'): phosephene_canny*cast , ord('v'): phosephene_canny_32*cast 
             }
            
            if key_press not in target.keys(): # If you press another keys 
                key_press = ord('1')


            zero_pad = np.pad(target[key_press], [(0,), (80,)], mode='constant')  # (480, 480) -> (480, 640) for showing in unity
                                                                            # 480 + 2*x = 640 => x = 80
            cv2.imshow("zero_padding (480, 640) for Unity", zero_pad)


            target_frame = zero_pad
            

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),80]  # 0 ~ 100 quality 
            #encode_param = [cv2.IMWRITE_PNG_COMPRESSION,0] # 0 ~ 9 Compressiong rate 
            #encode_param = [int(cv2.IMWRITE_WEBP_QUALITY),95]  # 0 ~ 100 quality 


            result, imgencode = cv2.imencode('.jpg', target_frame, encode_param)  # Encode numpy into '.jpg'
            data = np.array(imgencode)

            stringData = data.tostring()   # Convert numpy to string
            #print("byte Length: ", len(stringData))
            queue.put(stringData)          # Put the encode in the queue stack


     
            

        
    finally: 
        cv2.destroyAllWindows()

        # _Stop streaming 
        pipeline.stop()
        exit()

    



class MyTCPHandler(socketserver.BaseRequestHandler):

    queue  = enclosure_queue 
    stringData = str()

    def handle(self):
       
        # 'self.request' is the TCP socket connected to the client     
        print("A client connected by: ", self.client_address[0], ":", self.client_address[1] )


        while True:
            try:
                # _server <- client 
                self.data = self.request.recv(1024).strip()   # 1024 byte for header 
                #print("Received from client: ", self.data)

                if not self.data: 
                    print("The client disconnected by: ", self.client_address[0], ":", self.client_address[1] )     
                    break                

                # _Get data from Queue stack 
                MyTCPHandler.stringData = MyTCPHandler.queue.get()     

                # _server -> client 
                #print(str(len(MyTCPHandler.stringData)).ljust(16).encode())  # <str>.ljust(16) and encode <str> to <bytearray>
                
                ###self.request.sendall(str(len(MyTCPHandler.stringData)).ljust(16).encode())  # <- Make this line ignored when you connect with C# client. 
                self.request.sendall(MyTCPHandler.stringData)  

                
                #self.request.sendall(len(MyTCPHandler.stringData).to_bytes(1024, byteorder= "big"))
                #self.request.sendall(MyTCPHandler.stringData)             

                
            except ConnectionResetError as e: 
                print("The client disconnected by: ", self.client_address[0], ":", self.client_address[1] )     
                break


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
            exit()