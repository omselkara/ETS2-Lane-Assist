from keras import models
from Control import PC
import numpy as np
import cv2
import time
from PIL import ImageGrab
from Game import *

def advanced_edge_detection(image):
    maximum = np.max(image)
    multiplier = 255/maximum
    image = image*multiplier
    offset = 127.5-np.mean(image)
    image = image+offset
    image = image+image-127.5
    yöntem = (image//(160+np.std(image)/multiplier)).astype("uint8")
    yöntem = yöntem*255
    return yöntem


pc = PC()
delta = 0.0022968053817749023

def set_steering(x):
    global pc
    current = get_user_steer()
    pc.add_pos((x-current)/delta,0)

##model1 = models.load_model("corrected.keras")
##model2 = models.load_model("driver.keras")
model3 = models.load_model("model.keras")

def main():
    while True:
        #print(get_user_steer())
        pc.wait("y",500)
        while not pc.is_down("u"):
            if not is_cc_active():
                if get_speed_kmh()>=80 and get_speed_kmh()<81:
                    pc.press("c")
                    pc.key_up("w")
                else:
                    if get_speed_kmh()<80 and not pc.is_down("w"):
                        pc.key_down("w")
            img = np.array(ImageGrab.grab(bbox=screen_pos))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = advanced_edge_detection(img)
            img = cv2.resize(img,(335,95))/255
            data = get_all_data()
            inputs = np.array([img])
            output = model3.predict(inputs,verbose=0)[0][0]
##            print(get_user_steer(),output/100)
            set_steering(output/50)

main()

