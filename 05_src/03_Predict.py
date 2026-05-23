# IMPORT LIBRARIES

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # ignore this 
import os


# LOADING OUR TRAINED MODEL

model_path = '04_models/plant_model_v2_pro.keras'

if not os.path.exists(model_path):
    print(f" Error: Model file not found at {model_path}")
    exit()

model = tf.keras.models.load_model(model_path)

print("\nModel loaded successfully.\n")


# CLASS NAMES

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',

    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',

    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',

    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot'
]

def predict_my_plant(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return

    
    # LOAD AND PREPROCESS IMAGE :# 2. Image Preprocessing (AI ke size 150x150 aur scale mein convert karna)
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)        # Convert image to array
    img_array = img_array / 255.0                  # Normalize image 
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension


    # 3. Prediction (AI ka guess)
    prediction = model.predict(img_array)

    # 4. Find the best guess
    score = tf.nn.softmax(prediction[0])    # Convert the prediction values to probabilities
    result_index = np.argmax(score)
    result_label = class_names[result_index]    # Get class name
    confidence = 100 * np.max(score)        # Get confidence score



    print(f"\n-----------------------------------")

    print(f"Testing Image: {os.path.basename(img_path)}")
    print(f"AI Guess: '{result_label}'")
    print(f"Confidence Score: {confidence:.2f}%")

    print(f"-----------------------------------\n")


# EXECUTION (Test with a few photos)
# Path of photos :  
# predict_my_plant("03_Data/test_photos/image04.jpg")
predict_my_plant("02_dataset/03_test/0bbb8bce-2020-416b-8bd6-c160c2db9921___RS_Early.B 8386.JPG")
# predict_my_plant("03_Data/test_photos/images5.jpg")
# predict_my_plant("03_Data/test_photos/img1.jpg")