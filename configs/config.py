""" Model config in json format """


CFG = {
    "data": {
        "path": "./data/geological_similarity/",
        "label_mode": "categorical",
        "image_size": 32,
        "batch_size": 32,
        "validation_split": 0.2,
    },
    "train": {
        "epoches": 100
    },
    "base_model": {
        "input": [32, 32, 3],
        "layers_to_reuse": 15
    },
    "model": {
        "output": 6,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"],
        "loss": "categorical_crossentropy",
        "trained_models_path": "./trained_models/"
    }
}
