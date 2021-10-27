# Standard
import joblib

# External
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def run():
    # Data loading
    df = pd.read_csv("./dimensionality_reduction/reduced_data.csv")
    data = df.iloc[:,:-3]

    # KNN
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(data)

    # saving pca object for reuse
    joblib.dump(neigh, "./kneighbors_finding/kneighbors_model.joblib")
    print("6 neihbors model fitted with sucess!")




if __name__ == "__main__":
    run()