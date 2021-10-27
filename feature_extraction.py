# Standard
import os

# Data wrangling
import numpy as np
import pandas as pd

# Internal
from dataLoader.dataLoader import dataLoader
from models.model import FeatureExtractor
from configs.config import CFG



def extract_features_from_image(model, image):
    """ This function uses the feature extractor to extract features
     from the input image 
     """
    extract_features = model.predict(image)
    return extract_features


def extract_features_from_data(model):
    """ This function extracts features and generate a csv file """
    features = []
    classe = []
    image_name = []

    for classe_name in os.listdir(CFG["data"]["path"]):
        for img_name in os.listdir(CFG["data"]["path"] + classe_name):
            img = dataLoader().load_image(classe_name, img_name)
            img_preprocessed = dataLoader().preprocess_image(img)
            extracted_features = extract_features_from_image(model, img_preprocessed)
            extracted_features = extracted_features.reshape((64))
            features.append(extracted_features)
            classe.append(classe_name)
            image_name.append(img_name)

    df1 = pd.DataFrame(np.array(features))
    df2 = pd.DataFrame(np.array(classe))
    df3 = pd.DataFrame(np.array(image_name))

    df = pd.concat([df1, df2, df3], axis = 1)
    df.to_csv("./extracted_features/extracted_features.csv")
    print("features extracted  with sucess!")


if __name__ == "__main__":
    FeatureExtractorInstance= FeatureExtractor(CFG)
    FeatureExtractorInstance.load_model()
    FeatureExtractorModel = FeatureExtractorInstance.feature_extractor()
    extract_features_from_data(FeatureExtractorModel)