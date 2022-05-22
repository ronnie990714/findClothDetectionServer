import io
import json
import os
import shutil
from PIL import Image
import numpy as np

import torch

detected_list = list()
model_path = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')
img_input_dir = "img input path"
img_output_dir = "img output path"

# for testing byteArray file
with open("img input path",'rb') as f:
            data = f.read()
test_bin1 = io.BytesIO(data)

with open("img input path",'rb') as f:
            data = f.read()
test_bin2 = io.BytesIO(data)
# for testing byteArray file

#define export Model
def exportModel(img_input):
    # initialize
    detected_list.clear()
    i=1
    dict_name = "detected_result" + str(i)
    model = model_path
    model.eval()
    

    # if img_input is directory
    try: 
        input_image = os.listdir(img_input)
    
    # if img_input is byteArray
    except:
        # detect image - user sends new image to model
        deleteAllFiles(img_output_dir) # delete all files in output directory for new data input
        target = Image.open(img_input)
        results = model(target,size=640)
        data = results.pandas().xyxy[0].to_json(orient="index")
        data_json = json.loads(data)
        dict_name = dict()

        
        # save detected result
        for k in data_json:
            label = data_json[k]['name']
            target_area = (data_json[k]['xmin'], data_json[k]['ymin'], data_json[k]['xmax'], data_json[k]['ymax']) # find coordinate for crop target image
            detected_image_path = img_output_dir+"\\userImage"
            detected_image_name = detected_image_path + "\\" + label + ".jpeg" # cropped image is saved in img_output_path\userImage\[categoryName]
            if not os.path.isdir(detected_image_path): # if directory is not exist -> make directory
                os.mkdir(detected_image_path)
            target.crop(target_area).save(detected_image_name)
            dict_name[label]=detected_image_name
            return dict_name # if input is byteArray -> return type is dict
                             # output ex) {'T-shirt': 'img_path'}

    else: # if except does not accure -> img_input is directory
        for img_file in input_image:
            # detect image -> check file extension before run
            if(img_file.split(".").pop(1) == "jpg"):
                with open(img_input+"\\"+img_file,'rb') as f:
                    data = f.read()
                img_bytes = io.BytesIO(data)
                target_img = Image.open(img_bytes)
                results = model(target_img,size=640)
                data = results.pandas().xyxy[0].to_json(orient="index") 
                data_json = json.loads(data)
                dict_name = dict()

                # save detected result
                for k in data_json:
                    label = data_json[k]['name']
                    target_area = (data_json[k]['xmin'], data_json[k]['ymin'], data_json[k]['xmax'], data_json[k]['ymax']) # find coordinate for crop target image
                    detected_image_path = img_output_dir+"\\"+img_file.split(".").pop(0)
                    detected_image_name = detected_image_path + "\\" + label + ".jpeg"
                    if not os.path.isdir(detected_image_path):
                        os.mkdir(detected_image_path)
                    target_img.crop(target_area).save(detected_image_name)
                    dict_name[label]=detected_image_name
                detected_list.append(dict_name)
                i+=1

    return detected_list # if input is directory -> return type is list
                         # output ex) [{'Jeans': 'img_path', 'Coat': 'img_path'}, {'Pants': 'img_path', 'Hood T-shirt': 'img_path'}]


def deleteAllFiles(filePath):
    if(os.path.exists(filePath)):
        for file in os.scandir(filePath):
            shutil.rmtree(file.path)


print(exportModel(test_bin1)) # for testing - byteArray file 1   
print(exportModel(img_input_dir)) # for testing - image file
# print(exportModel(test_bin2)) # for testing - byteArray file 2   

