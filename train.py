""" train.py """

# internal
from configs.config import CFG
from models.model import FeatureExtractor
from utils.plots import Plot


def run():
    """Builds model, loads data, trains and save model"""
    model = FeatureExtractor(CFG)
    model.load_data()
    model.build()
    model.train()



if __name__ == '__main__':
    run()
