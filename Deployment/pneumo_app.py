import streamlit as st
import numpy as np
import pandas as pd
import os, urllib, cv2
from PIL import Image, ImageDraw
from PIL import ImagePath
from pathlib import Path
import zipfile
import tensorflow as tf
from model import *
from skimage import exposure
import requests
    
def main():  
    readme_text = st.markdown(get_file_content_as_string("readme.md"))
    st.sidebar.title("What to do")
    
     # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    unzipData("dicom-jpg.zip")  
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("pneumo_app.py"))

def run_the_app():

    @st.cache
    def loaddata(url):
        return pd.read_csv(url)
     
    
    train_data = loaddata(DATA_URL_PATH + "/train.csv")
    st.write('## Data extracted from dicom images', train_data)
    
    train_rle = loaddata(DATA_URL_PATH + "/train-rle.csv")
    train_rle.columns = ['ImageId', 'EncodedPixels']
    
    selected_frame_index = st.sidebar.selectbox("Choose an data frame index", [i + 1 for i in range(1000)])
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return
    
    imagepath = get_image_path(train_data, selected_frame_index - 1)
    image = tf.keras.preprocessing.image.load_img(imagepath)

    info = "Selected Datapoint: " + str(selected_frame_index)
    st.subheader(info)
    info = "Patient's Age: " +  str(train_data.iloc[selected_frame_index]['age'])
    st.write(info)
    info = "Patient's Sex: " +  str(train_data.iloc[selected_frame_index]['sex'])
    st.write(info)
    info = "Has Pneumothorax? " +  "Yes" if train_data.iloc[selected_frame_index]['has_pneumothorax'] == 1 else "No"
    st.write(info)
    info = "X-Ray"
    st.write(info)
    st.image(image, use_column_width=True)
    
    
    encodedPixels = train_rle[train_rle['ImageId'] == train_data.iloc[selected_frame_index]['ImageId']]['EncodedPixels'].values
    mask = np.zeros((1024, 1024))
    for pix in encodedPixels:
        mask = mask + rle2mask(pix, 1024, 1024).T
    info = "Original Mask"
    st.write(info)
    st.image(mask.astype(np.uint8), use_column_width=True)
    
    
    img_size = 256
    xrayimage = tf.keras.preprocessing.image.load_img(imagepath)
    xrayimage = tf.keras.preprocessing.image.img_to_array(xrayimage, dtype='float32')  
    xrayimage = tf.image.resize(xrayimage, [img_size, img_size])
    xrayimage = xrayimage / 255.0
    xrayimage = exposure.equalize_adapthist(xrayimage)
    xrayimage = tf.expand_dims(xrayimage, axis = 0)
    
    #Model1
    threshold = 0.38
    predictor = buildPredictor()
    predictor.load_weights("best_model1.h5")
    
    classifier = buildClassifier()
    classifier.load_weights("best_model3.h5")

    is_pneumothorax = classifier.predict(xrayimage) 
    is_pneumothorax = is_pneumothorax[0]
   
    if is_pneumothorax >= threshold :
        pred_masks = predictor.predict(xrayimage)
        pred_masks = pred_masks[0]
    else :
        pred_masks = np.zeros((256,256,1))
    
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_masks)
    info = "Predicted Mask : 1"
    st.subheader(info)
    info = "This model was trained on only pneumothorax dataset along with a classifer"
    st.write(info)
    st.image(pred_mask, use_column_width=False)
    
    #Model2
    classifier_over = buildClassifier()
    classifier_over.load_weights("best_model4.h5")

    is_pneumothorax = classifier_over.predict(xrayimage) 
    is_pneumothorax = is_pneumothorax[0]
   
    if is_pneumothorax >= threshold :
        pred_masks = predictor.predict(xrayimage)
        pred_masks = pred_masks[0]
    else :
        pred_masks = np.zeros((256,256,1))
    
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_masks)
    info = "Predicted Mask : 2"
    st.subheader(info)
    info = "This model was trained on only pneumothorax dataset along with a classifer with wrongly classified being oversampled"
    st.write(info)
    st.image(pred_mask, use_column_width=False)
    
    #Model3 
    is_pneumothorax = classifier.predict(xrayimage) 
    is_pneumothorax = is_pneumothorax[0]
    
    if is_pneumothorax >= threshold :
        interpreter_predictor = tf.lite.Interpreter(model_path=str("predictor.tflite"))
        interpreter_predictor.allocate_tensors()
        input_index = interpreter_predictor.get_input_details()[0]["index"]
        output_index = interpreter_predictor.get_output_details()[0]["index"]
        print(input_index)
        
        interpreter_predictor.set_tensor(input_index, xrayimage.numpy())
        interpreter_predictor.invoke()
        pred_masks = tf.reshape(interpreter_predictor.get_tensor(output_index), [256, 256, 1])
        
    else :
        pred_masks = np.zeros((256,256,1))
    
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_masks)
    info = "Predicted Mask : 3"
    st.subheader(info)
    info = "This model was trained on only pneumothorax dataset along with a classifer with wrongly classified being oversampled"
    st.write(info)
    st.image(pred_mask, use_column_width=False)
    
    #Model4
    is_pneumothorax = classifier_over.predict(xrayimage) 
    is_pneumothorax = is_pneumothorax[0]
    
    if is_pneumothorax >= threshold :
        interpreter_predictor = tf.lite.Interpreter(model_path=str("predictor.tflite"))
        interpreter_predictor.allocate_tensors()
        input_index = interpreter_predictor.get_input_details()[0]["index"]
        output_index = interpreter_predictor.get_output_details()[0]["index"]
        print(input_index)
        
        interpreter_predictor.set_tensor(input_index, xrayimage.numpy())
        interpreter_predictor.invoke()
        pred_masks = tf.reshape(interpreter_predictor.get_tensor(output_index), [256, 256, 1])
        
    else :
        pred_masks = np.zeros((256,256,1))
    
    pred_mask = tf.keras.preprocessing.image.array_to_img(pred_masks)
    info = "Predicted Mask : 4s"
    st.subheader(info)
    info = "This model was trained on only pneumothorax dataset along with a classifer with wrongly classified being oversampled"
    st.write(info)
    st.image(pred_mask, use_column_width=False)
      

def unzipData(zipPath):
    with zipfile.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.printdir()
        zip_ref.extractall()

  
def get_image_path(train_data, data_index):
    imagepath = train_data.iloc[data_index]['images_paths']
    imagepath = imagepath.split('/')[-1]
    imagepath = os.path.join(os.getcwd() , "dicom-jpg", imagepath)  
    return imagepath

def download_file(file_path):
    
    os_path = Path(file_path)
    
    info = "Downloading file: " + file_path + "... this may take a while ! \n Don't stop it!"
    if not os_path.exists():
        with st.spinner(info):
            from google_drive_download import download_file_from_google_drive
            download_file_from_google_drive(EXTERNAL_DEPENDENCIES[file_path]["id"], os_path)
            
       
def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
 
        
def get_file_content_as_string(path):    
    url = DATA_URL_PATH + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


DATA_URL_PATH = "https://raw.githubusercontent.com/amarshrma/Pneumothorax-Detection/master/Deployment/"
# External files to download.
EXTERNAL_DEPENDENCIES = {
    "predictor_weights.h5": {
        "id": "1gQXexxr_6F04Kbm0Gdh4JW9AyNiICy5x"
    },
    "classifier_weights.h5": {
        "id" : "1-EQA215dmoPFPXomGSi9dXFifVaSaa-F"
    },
    "classifier_oversampled_weights.h5" : {
        "id" : "1dyK3wiP5u2XnPCONHekk7_yn_KHOaoS6"
    },
    "dicom-jpg.zip": {
        "id": "116_OHC7lPUYwpIUh6jRcU-lPZCSzt0dR"
    },
    "classifier.tflite": {
        "id": "1O_-yVo9ebd02FvaXnxH6ohqKaCtwoiRt"
    },
    "classifier_over.tflite": {
        "id": "1KJIl01KUrjBhMTs65DyLTv8khapOVie0"
    },
    "predictor.tflite": {
        "id": "18kltWQOySogj-KUByoGs_ISYhs8Np9pU"
    }
    
}
if __name__ == "__main__":
    main()