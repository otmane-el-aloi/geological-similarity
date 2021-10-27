import pandas as pd
import joblib
from dataLoader.dataLoader import dataLoader
from models.model import FeatureExtractor
from configs.config import CFG
import matplotlib.pyplot as plt
import tensorflow as tf


# Reduced Data directory
REDUCED_DATA_PATH = "dimensionality_reduction/reduced_data.csv"

# Upload directory
UPLOAD_DIRECTORY = "static/images"

# Static data directory
STATIC_DATA_Directory = "static/images"

# KNeighbors model directory 
KNEIGHBORS_DIRECTORY = "kneighbors_finding/kneighbors_model.joblib"

# PCA model 
PCA_MODEL_DIRECTORY = "dimensionality_reduction/pca_model.joblib"

# Load data
df = pd.read_csv(REDUCED_DATA_PATH)
df["class"] = df["class"].astype("str")


def extract_features(img):
    """ This function takes an image of interest, extracts feautres and reduces its dimension """
    # load models
    model = FeatureExtractor(CFG)
    model.load_model()
    feature_extractor = model.feature_extractor()

    # extract features 
    print(type(img))
    extracted_features = feature_extractor.predict([img])

    # reduce dimension
    pca_model = joblib.load(PCA_MODEL_DIRECTORY)
    reduced_img = pca_model.transform(extracted_features)
    return reduced_img


def run():
    data_loader_object = dataLoader()
    neighrest_images = []
    neighrest_classes = []
    neighrest_directories = []

    # Loading the uploaded image and performing feature extraction
    image_of_interest = data_loader_object.load_uploaded_img(UPLOAD_DIRECTORY, "image_of_interest.jpg")
    preprocessed_img = data_loader_object.preprocess_image(image_of_interest)
    reduced_image = extract_features(preprocessed_img)

    # Finding the 6 neighrest images 
    neigh = joblib.load(KNEIGHBORS_DIRECTORY)
    reduced_image = reduced_image.reshape((3))
    print("reduced_img: {}".format(reduced_image))
    neighbors = neigh.kneighbors([reduced_image], return_distance=False)
    print(neighbors)
    for index_neighbor in neighbors[0]:
        neighbor_name = df.loc[index_neighbor, "name"]
        neighbor_class = df.loc[index_neighbor, "class"]
        neighrest_images.append(neighbor_name)
        neighrest_classes.append(neighbor_class)


if __name__== "__main__":
    run()