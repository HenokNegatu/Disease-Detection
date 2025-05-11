import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import inspect
# =========================
# 🔹 STEP 1: Set your paths here
# =========================

# 👉 Path to your trained model (change this if saved somewhere else)
MODEL_PATH = "/home/henok/Documents/girume/api/src/lung_model.h5"

# 👉 Path to your test X-ray image (change this to the image you want to test)
TEST_IMAGE_PATH = "/home/henok/Documents/girume/api/src/images.jpeg"  # example: "sample_images/patient1.png"

# Class labels (must be in the same order as your training script used)
class_names = ['covid', 'normal', 'pneumonia', 'viral_pneumonia']


class PredictModel:
    # =========================
    # 🔹 STEP 2: Load and preprocess image
    # =========================
    async def prepare_image(self, uploaded_file):
        file_read_result = uploaded_file.read()
        
        if inspect.isawaitable(file_read_result):
            file_bytes = await file_read_result
        else:
            file_bytes = file_read_result
        
        file_stream = io.BytesIO(file_bytes)
        img = image.load_img(file_stream, target_size=(150, 150)) 
        img_array = image.img_to_array(img)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=int(0))

        print("Type of img_array:", type(img_array))
        print("Shape of img_array:", img_array.shape)
        
        return img_array

    # def prepare_image(self, img_path):
    #     img = image.load_img(img_path, target_size=(150, 150))  # must match training size
    #     img_array = image.img_to_array(img)
    #     img_array = img_array / 255.0  # Normalize
    #     img_array = np.expand_dims(img_array, axis=0)  # Make it batch shape
    #     return img_array


    # =========================
    # 🔹 STEP 3: Prediction logic
    # =========================
    async def predict_image(self, model_path, img_path):
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return

        # if not os.path.exists(img_path):
        #     print(f"❌ Image file not found at: {img_path}")
        #     return

        print("✅ Loading trained model...")
        model = load_model(model_path)

        print("📸 Preprocessing image...")
        img_array = await self.prepare_image(img_path)

        print("🤖 Making prediction...")
        predictions = model.predict(img_array)[0]
        print("Predictions type:", type(predictions))
        print("Predictions shape:", predictions.shape)

        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = predictions[predicted_index]

        return {
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "class_scores": predictions.tolist()
        }

        # # Show the image with the predicted label
        # img = image.load_img(img_path, target_size=(150, 150))
        # plt.imshow(img)
        # plt.title(f"Prediction: {predicted_label} ({confidence:.2f})")
        # plt.axis("off")
        # plt.show()


# # =========================
# # 🔹 STEP 4: Run the prediction
# # =========================
# if __name__ == "__main__":
#     model = PredictModel()
#     img_array = model.predict_image(MODEL_PATH, TEST_IMAGE_PATH)
#     model.predict_image(img_array)
