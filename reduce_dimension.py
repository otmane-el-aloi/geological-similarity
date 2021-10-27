# Standard
import joblib

# Data 
import pandas as pd
from sklearn.decomposition import PCA



def run():
# Data loading
    df = pd.read_csv("./extracted_features/extracted_features.csv")

    # Dimensionn reduction
    pca  = PCA(3)
    pca.fit(df.iloc[:,:-3])


    reduced_data = pca.transform(df.iloc[:,:-3])
    reduced_df = pd.DataFrame(reduced_data, columns=["PC1", "PC2", "PC3"])

    id_features = df.iloc[:,[-2,-1]]
    id_features.columns = ["class", "name"]
    reduced_df = pd.concat([reduced_df, id_features], axis = 1)

    # Saving reduced data
    reduced_df.to_csv("./dimensionality_reduction/reduced_data.csv")

    # saving pca object for reuse
    joblib.dump(pca, "./dimensionality_reduction/pca_model.joblib")

    print("dimension reduction with success!")




if __name__ == "__main__":
    run()