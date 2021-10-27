""" loading  and preprocessing  images from local folders"""
# Image processing
import cv2
import tensorflow as tf
import numpy as np

# Internal
from configs.config import CFG
import matplotlib.pyplot as plt 




class dataLoader:
    def __init__(self):
        self.data_path = CFG["data"]["path"]
        self.image_size = CFG["data"]["image_size"]
        
    def load_image(self, classe, image_name):
        """ Loads an image given it's classe and it's name"""
        img = cv2.imread(self.data_path + classe + "/" + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_uploaded_img(self, upload_directory, image_name):
        """ This method loads an uploaded image from the uploading directory"""
        img = cv2.imread(upload_directory + "/" + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def preprocess_image(self, img):
        """ Preprocess an image for the feature extractor"""
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        # img_normalized = (img_resized/255).astype(np.uint8)
        
        # plt.imshow(img_normalized)
        # plt.show()
        img_normalized = tf.expand_dims(img_resized, axis = 0)
        return img_normalized


