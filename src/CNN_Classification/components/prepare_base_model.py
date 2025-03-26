import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from src.CNN_Classification.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        """Loads the VGG16 base model with pre-trained weights."""
        self.model = tf.keras.applications.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(self.config.base_model_path, self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """Adds custom layers on top of the VGG16 model."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None and freeze_till > 0:
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Add custom layers
        flatten_in = tf.keras.layers.Flatten()(model.output)
        dropout = tf.keras.layers.Dropout(0.3)(flatten_in)
        dense = tf.keras.layers.Dense(128, activation="relu")(dropout)
        dropout = tf.keras.layers.Dropout(0.2)(dense)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(dropout)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["sparse_categorical_accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        """Updates the base model by adding classification layers."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(self.config.updated_base_model_path, self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Saves the model to the given path."""
        model.save(f"{path}")
