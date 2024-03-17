import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model

def predict_file(file):
    image = Image.open(file)
    image = image.resize((224,224))
    st.image(image)
    image = np.array(image)
    image = image/255.0
    image = image.reshape(1,224,224,3)
    prediction = model.predict(image)
    return prediction
    


model = load_model('TLDC1.h5')
classes = {0: 'Tomato_Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Late_blight',\
          3: 'Tomato_Leaf_Mold', 4: 'Tomato_Septoria_leaf_spot', 5: 'Tomato_Spider_mites_Two_spotted_spider_mite',\
          6: 'Tomato_Target_Spot', 7: 'Tomato_Tomato_YellowLeaf_Curl_Virus',\
          8: 'Tomato_Tomato_mosaic_virus', 9: 'Tomato_healthy'}

st.header('RCEE::AI&DS')
st.title('TOMATO LEAF DISEASE CLASSIFICATION')

file = st.file_uploader('Upload any Tomato leaf')
if file:
    c = predict_file(file)
    st.text('Predicted as ' + classes[np.argmax(c)])

