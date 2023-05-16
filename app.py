import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow import keras
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def import_and_predict(image, model):
  image = np.asarray(image)
	
  image = image/255.
  img_resize= image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
  prediction = model.predict(img_resize)
  return prediction

def main():

  st.title("Image Noise Reduction")
  file = st.file_uploader("Please upload an image file", type=["jpg","png"])
  print("load the module")
  model = load_model("model.h5", compile=False)
  model.load_weights('model.h5')

  #
  if file is None:
      st.text("No image file is uploaded")
  

  else:
      
      image = Image.open(file)
     
      prediction = import_and_predict(image, model)
      st.text("Original Image")
      st.image(image, use_column_width=True)
      st.text("Denoised Image")
      st.image(prediction[0,:,:,:], use_column_width=True)

    
if __name__=='__main__':
    main()