# import tensorflow as tf
# import numpy as np
# import sys

# # Path to your saved model
# MODEL_PATH = "plant_disease_model.keras"
# IMAGE_SIZE = (160, 160)  # Must match training

# # Load trained model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Update with your actual class names (from training dataset)
# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#     'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
#     'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
#     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#     'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
# ]

# # Check if image path is given
# if len(sys.argv) < 2:
#     print("Usage: python pred_disease.py <image_path>")
#     sys.exit(1)

# img_path = sys.argv[1]

# # ✅ Load and preprocess image (force RGB, 3 channels always)
# img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE, color_mode="rgb")
# img_array = tf.keras.utils.img_to_array(img)  # shape (160,160,3)
# img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize to (1,160,160,3)

# # Predict
# pred = model.predict(img_array)
# pred_class_idx = np.argmax(pred, axis=1)[0]
# pred_class = class_names[pred_class_idx]
# confidence = float(np.max(pred) * 100)

# print(f"Predicted class index: {pred_class_idx}")
# print(f"Predicted disease: {pred_class}")
# print(f"Confidence: {confidence:.2f}%")
