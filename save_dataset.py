from Control import PC
from PIL import Image,ImageGrab
import numpy as np
import time
from Game import *
import cv2
from winsound import Beep

pc = PC()

imgs = []
datas = []
try:
    file = open("dataset/data.txt","r")
    data = file.readlines()
    file.close()
    count = len(data)
    print(count)
except:
    count = 0
    file = open("dataset/data.txt","w")
    file.close()
try:
    while True:
        pc.wait("q",500)
        file = open("dataset/data.txt","a")
        while not pc.is_down("p"):
            img = ImageGrab.grab(bbox=screen_pos)
            inputs = []
            if pc.is_down("w"):
                inputs.append(1)
            else:
                inputs.append(0)
            if pc.is_down("s"):
                inputs.append(1)
            else:
                inputs.append(0)
            datas.append(get_all_data()+inputs)
            imgs.append(img)
            if len(imgs)==100:
                for i in range(len(imgs)):
                    img = np.array(imgs[i])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = Image.fromarray(img)
                    img.save(f"dataset/images/image{count}.png")
                    for j in datas[i]:
                        file.write(str(j)+" ")
                    file.write("\n")
                    count += 1
                print(f"{count} images")
                imgs = []
                datas = []
                
            else: 
                time.sleep(0.1)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = Image.fromarray(img)
            img.save(f"dataset/images/image{count}.png")
            for j in datas[i]:
                file.write(str(j)+" ")
            file.write("\n")
            count += 1
        print(f"{count} images")
        file.close()
except Exception as e:
    pc = e
    print(pc)
    Beep(1000,300)
    Beep(1000,300)
    Beep(1000,300)
    Beep(1000,300)
    file.close()
    
