# Importing essential libraries and modules

from flask import Flask, render_template, request,Response, render_template_string,redirect,url_for
from markupsafe import Markup
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model, Sequential
# import Image
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
import os
# import seaborn as sns
# sns.set_style('darkgrid')
from PIL import Image

# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==============================================================================================

model_path="EfficientNetB3-skindisease-83.00.h5"
model=load_model(model_path)


def predictor(sdir, csv_path,  crop_image = False):    
    # read in the csv file
    class_df=pd.read_csv(csv_path,encoding='cp1252')    
    # img_height=int(class_df['height'].iloc[0])
    img_height=int(class_df['width'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)
    scale=1
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1]) 
        print (s1,s2)
    path_list=[]
    paths=sdir
    # print('path',paths)
    # for f in paths:
    path_list.append(paths)
    image_count=1    
    index_list=[] 
    prob_list=[]
    cropped_image_list=[]
    good_image_count=0
    for i in range (image_count):
               
        img=cv2.imread(path_list[i])
        # print('i',img)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop_image == True:
            status, img=crop(img)
        else:
            status=True
        if status== True:
            good_image_count +=1
            img=cv2.resize(img, img_size)            
            cropped_image_list.append(img)
            img=img*s2 - s1
            img=np.expand_dims(img, axis=0)
            p= np.squeeze (model.predict(img))           
            index=np.argmax(p)            
            prob=p[index]
            index_list.append(index)
            prob_list.append(prob)
    if good_image_count==1:
        # print(class_df.columns.tolist())
        class_name= class_df['class'].iloc[index_list[0]]
        symtom=class_df['symtoms '].iloc[index_list[0]]
        # symtom='1'
        medicine=class_df['medicine'].iloc[index_list[0]]
        wht=class_df['what is'].iloc[index_list[0]]
        probability= prob_list[0]
        img=cropped_image_list [0] 
        # plt.title(class_name, color='blue', fontsize=16)
        # plt.axis('off')
        # plt.imshow(img)
        # print(symtom,medicine,wht)
        return class_name, probability,symtom,medicine,wht
    elif good_image_count == 0:
        return None, None,None,None,None
    most=0
    for i in range (len(index_list)-1):
        key= index_list[i]
        keycount=0
        for j in range (i+1, len(index_list)):
            nkey= index_list[j]            
            if nkey == key:
                keycount +=1                
        if keycount> most:
            most=keycount
            isave=i             
    best_index=index_list[isave]    
    psum=0
    bestsum=0
    for i in range (len(index_list)):
        psum += prob_list[i]
        if index_list[i]==best_index:
            bestsum += prob_list[i]  
    img= cropped_image_list[isave]/255    
    class_name=class_df['class'].iloc[best_index]
    symtom=class_df['symtoms '].iloc[best_index]
    medicine=class_df['medicine'].iloc[best_index]
    wht=class_df['what is'].iloc[best_index]
    # print(+symtom,medicine,wht)
    # plt.title(class_name, color='blue', fontsize=16)
    # plt.axis('off')
    # plt.imshow(img)
    return class_name, bestsum/image_count,symtom,medicine,wht
img_size=(300, 300)

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


import base64
def render_picture(data):

    render_pic = base64.b64encode(data).decode('ascii') 
    return render_pic

@app.route('/disease')
def disease():
    return render_template('disease.html')


@app.route('/disease-predict', methods=['POST','GET' ])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    if request.method:
        if request.method == 'POST':
            img1 = request.files['file1']
           
                
        else:
            img1 = request.args.get('file1')
         
        img1.save("out.jpg")   
              
        path="out.jpg"
        
        csv_path="class.csv"
        model_path="EfficientNetB3-skindisease-83.00.h5"
        class_name, probability,symtom,medicine,wht=predictor(path, csv_path, crop_image = False)
       
        print(symtom)

        prediction = Markup(class_name)
            
        
        return render_template('disease-result.html', prediction=prediction,symtom=symtom,medicine=medicine,wht=wht, title=title)


  

if __name__ == '__main__':
    app.run(debug=True)
