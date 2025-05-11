import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import numpy as np


class MobileNetV2Model:
    def __init__(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = Flatten()(x)
        x = Dense(6, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=x)

    def predict(self, img_array):
        # Convert to float32 if not already
        img_array = img_array.astype(np.float32)
        # Apply MobileNetV2 specific preprocessing
        img_array = preprocess_input(img_array)
        predictions = self.model.predict(img_array)
        print(predictions)
        return predictions

