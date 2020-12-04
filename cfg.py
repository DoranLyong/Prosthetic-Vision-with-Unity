exp_cfg = {
    "imgShape" : (480, 480),
    "pixSize" : (48, 48), 
    "Strength" : 0.8,
    "hed_pretrained" : "./model/data/hed_pretrained_bsds.caffemodel",
    "prototxt" : "./model/data/deploy.prototxt"
}


"""
1. Don't change 480x480 because 'D435' camera only read images in 640x480.

2. 'pixSize' should be common division of the 'imgShape'. 
    ex) 8x8, 20x20, 32x32, 40x40 with regard to 480x480 image. 
"""