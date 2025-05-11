import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import MobileNetV2

# =========================
# 🔹 STEP 1: Set your paths here
# =========================
MODEL_PATH = "/home/henok/Documents/girume/api/src/lung_model.h5"
TEST_IMAGE_PATH = "/home/henok/Documents/girume/api/src/1278.png"

# Class labels (MUST match training order)
class_names = ['bacterial_pneumonia', 'cardiomegaly', 'covid', 'normal', 'tuberculosis', 'viral_pneumonia']

# =========================
# 🔹 STEP 2: Load and preprocess image
# =========================
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# 🔹 STEP 3: Prediction logic
# =========================
def predict_image(model_path, img_path):
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return
    if not os.path.exists(img_path):
        print(f"❌ Image file not found at: {img_path}")
        return

    print("✅ Loading trained MobileNetV2 model...")
    model = load_model(model_path)
    # model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    print("📸 Preprocessing image...")
    img_array = prepare_image(img_path)

    print("🤖 Making prediction...")
    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[predicted_index]

    print("\n🎯 Prediction Results:")
    print(f"➡️ Predicted Class: {predicted_label} ({confidence:.2f})")
    print("📊 Class Confidence Scores:")
    for i, prob in enumerate(predictions):
        print(f"   {class_names[i]}: {prob:.4f}")

    # Display image with prediction
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f})")
    plt.axis("off")
    plt.show()

# =========================
# 🔹 STEP 4: Run prediction
# =========================
if __name__ == "__main__":
    predict_image(MODEL_PATH, TEST_IMAGE_PATH)
