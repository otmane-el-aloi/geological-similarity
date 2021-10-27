""" Transfer learning using VGG16 as a base model"""

# standard library

# external
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


class BaseModelVGG16:
    """Base Model class"""

    def __init__(self, cfg):
        self.config = cfg
        self.base_model = VGG16(input_shape=self.config["base_model"]["input"],
                                include_top=False)

    def reuseLayers(self):
        """Sets firsts layers as untrainable"""
        count = 1
        for layer in self.base_model.layers:
            if count <= self.config["base_model"]["layers_to_reuse"]:
                layer.trainable = False
            count += 1

    def getBaseModel(self):
        """ Returns backbone Model"""
        self.reuseLayers()
        return self.base_model


class FeatureExtractor:
    """ Feature Extractor Model Class """

    def __init__(self, cfg):
        self.config = cfg

        # base model
        self.base_model = BaseModelVGG16(cfg).getBaseModel()
        self.model = None

        # model training/aarchitecture params
        self.output_channels = self.config["model"]["output"]
        self.optimizer = self.config["model"]["optimizer"]["type"]
        self.metrics = self.config["model"]["metrics"]
        self.image_size = self.config["data"]["image_size"]
        self.batch_size = self.config["data"]["batch_size"]
        self.val_split = self.config["data"]["validation_split"]
        self.epoches = self.config["train"]["epoches"]
        self.loss = self.config["model"]["loss"]

        # data
        self.data_path = self.config["data"]["path"]
        self.train_dataset = None
        self.validation_dataset = None

        # Trained model path 
        self.model_path = self.config["model"]["trained_models_path"]

    def load_data(self):
        """Loads and preprocess data in batches"""
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(self.data_path,
                                                                         labels='inferred',
                                                                         label_mode=self.config["data"]["label_mode"],
                                                                         class_names=None,
                                                                         batch_size=self.config["data"]["batch_size"],
                                                                         image_size=(
                                                                             self.image_size, self.image_size),
                                                                         shuffle=True,
                                                                         seed=100,
                                                                         validation_split=self.val_split,
                                                                         subset='training')

        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.data_path,
                                                                              labels='inferred',
                                                                              label_mode=self.config["data"]["label_mode"],
                                                                              class_names=None,
                                                                              batch_size=self.config["data"]["batch_size"],
                                                                              image_size=(
                                                                                  self.image_size, self.image_size),
                                                                              shuffle=True,
                                                                              seed=100,
                                                                              validation_split=self.val_split,
                                                                              subset='validation')
        # self.preprocess_data()


    def preprocess_data(self):
        """ Normalises images"""
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
            1/255)
        self.train_dataset = self.train_dataset.map(
            lambda x, y: (normalization_layer(x), y))
        self.validation_dataset = self.validation_dataset.map(
            lambda x, y: (normalization_layer(x), y))

    def build(self):
        """ Builds the Keras model based """
        # add new classifier layers
        x = tf.keras.layers.Flatten()(self.base_model.layers[-1].output)
        latent_space = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(
            self.output_channels, activation='softmax')(latent_space)

        # define new model
        self.model = Model(inputs=self.base_model.inputs, outputs=output)

    def train(self):
        """Compiles and trains the model"""

        # compiling the model
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        # Early Stopping
        EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0, 
                                                         patience=3,
                                                         verbose=0,
                                                         mode='auto',
                                                         baseline=None,
                                                         restore_best_weights=False)
        # training the model
        model_history = self.model.fit(self.train_dataset, epochs=self.epoches,
                                       validation_data=self.validation_dataset,
                                       callbacks=[EarlyStopping])
        # saving the trained model
        self.save()
        return model_history.history['loss'], model_history.history['val_loss']

    def feature_extractor(self):
        """ Creates the feature extractor model"""
        feature_extractor = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-2].output)
        print("feature extractor created with success!")
        return feature_extractor

    def load_model(self):
        """ loades the model to the model attribute"""
        self.model = tf.keras.models.load_model(self.model_path + "feature_extractor.h5")
        print("model loaded with success!")

    def save(self):
        """ saves the model in h5 format"""
        self.model.save(self.config["model"]["trained_models_path"] +
                        "feature_extractor.h5")
        print("saving model done with sucess!")
