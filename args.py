import argparse
import os.path as osp 
from collections import OrderedDict



def argument_parser():
    ws_dir = osp.dirname(osp.abspath(__file__))

    ap = argparse.ArgumentParser() 

    ap.add_argument('-i', "--input", type=str, default=osp.join(ws_dir,'..', 'data', 'videos', 'example_01.mp4'),
                     help="path to input video file")

    ap.add_argument('-s', "--skip-frames", type=int, default=20, 
                     help="the number of skip frames between detections")

    # *****************************
    # SSD model 
    # *****************************
    ap.add_argument("-p", "--prototxt", type=str, default=osp.join(ws_dir, 'model', 'mobilenet_ssd', 'MobileNetSSD_deploy.prototxt'),
                 help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", type=str, default=osp.join(ws_dir, 'model', 'mobilenet_ssd', 'MobileNetSSD_deploy.caffemodel'),
                 help="path to Caffe pre-trained model")

    ap.add_argument("-pt", "--pbtxt", type=str, default=osp.join(ws_dir, 'model', 'mobilenet_ssd', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'),
                 help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-pb", "--pb", type=str, default=osp.join(ws_dir, 'model', 'mobilenet_ssd', 'frozen_inference_graph.pb'),
                 help="path to Caffe pre-trained model")

    ap.add_argument('-c', "--confidence", type=float, default=0.45
                ,help="minimum probability to filter weak detections")



    return ap 





def SSD_classes():
    # _Start: initialize the list of class labels MobileNet SSD was trained to detect 
#    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor"]    
    
    CLASSES = OrderedDict()  # https://www.daleseo.com/python-collections-ordered-dict/
    
    """
    (ref): https://stackoverflow.com/questions/4803999/how-to-convert-a-file-into-a-dictionary
    """
    with open("coco_labels.txt") as f: 
        for line in f: 
            key, val = line.split()        
            CLASSES[int(key)] = val


    return CLASSES



""" coco_labels.txt 

[ref] https://gist.github.com/aallan/fbdf008cffd1e08a619ad11a02b74fa8


"""