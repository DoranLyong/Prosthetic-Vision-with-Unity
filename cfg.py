exp_cfg = {
    "imgShape" : (640, 640),
    "pixSize" : (32,32), 
    "Strength" : 3,
    "hed_pretrained" : "./model/data/hed_pretrained_bsds.caffemodel",
    "prototxt" : "./model/data/deploy.prototxt"
}


"""
'pixSize' should be common division of the 'imgShape'. 

ex) 8x8, 20x20, 32x32, 40x40 with regard to 640x640 image. 
"""