import cv2
import numpy as np

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

def load_outs(path):
    outputs = []

    file = open(path,"r")
    data = file.readlines()
    file.close()

    interval = len(data)//500
    if interval==0:
        interval = 1
    for i in range(len(data)):
        outs = data[i].rstrip(" \n").split(" ")
        outputs.append(outs)

    #outputs = np.array(outputs)
    return outputs


asd = 0
def save_images(path,start_index,outs,file):
    global asd
    for i in range(len(outs)):
        img = cv2.imread(f"{path}/image{i}.png", cv2.IMREAD_GRAYSCALE)
        img = advanced_edge_detection(img)
        img = cv2.resize(img,(335,95))
        cv2.imwrite(f'dataset/images/image{start_index+i}.png', img)
        for j in outs[i]:
            file.write(str(j)+" ")
        file.write("\n")

outs1 = load_outs("dataset large/data.txt")
#outs2 = load_outs("dataset driver/data.txt")
#outs3 = load_outs("dataset en son/data.txt")
#print(len(outs1),len(outs2),len(outs3))

file = open("dataset/data.txt","w")

print("birinci")
save_images("dataset large/images",0,outs1,file)
##print("ikinci")
##save_images("dataset driver/images",len(outs1),outs2,file)
##print("üçüncü")
##save_images("dataset en son/images",len(outs1)+len(outs2),outs3,file)

file.close()


