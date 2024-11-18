import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pickle  # Assuming TensorFlow/Keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the pre-trained model
model = pickle.load(open('finalized_model.pkl', 'rb'))

# Function to decode prediction into a disease, solution, and link
def decode_prediction(prediction):
    mapping = {
        0: {"disease": "Actinic keratosis", 
            "solution": "Use sunscreen and avoid excessive sun exposure. Consult a dermatologist for cryotherapy or photodynamic therapy.",
            "link": "https://www.cdc.gov/actinic_keratosis"},
        1: {"disease": "Atopic Dermatitis", 
            "solution": "Moisturize regularly, avoid irritants, and use prescribed topical corticosteroids.",
            "link": "https://www.nhs.uk/conditions/atopic-eczema/"},
        2: {"disease": "Benign keratosis", 
            "solution": "No treatment required, but cryotherapy or laser treatment can be done for cosmetic reasons.",
            "link": "https://www.aad.org/public/diseases/keratoses"},
        3: {"disease": "Dermatofibroma", 
            "solution": "No treatment required unless painful. Surgical removal can be an option.",
            "link": "https://www.aad.org/public/diseases/dermatofibromas"},
        4: {"disease": "Melanocytic nevus", 
            "solution": "Most are harmless but regular monitoring is recommended. Seek medical advice if changes occur.",
            "link": "https://www.cancer.org/cancer/melanoma-skin-cancer.html"},
        5: {"disease": "Melanoma", 
            "solution": "Seek immediate medical advice for potential surgery or targeted therapy.",
            "link": "https://www.cancer.gov/types/skin/melanoma"},
        6: {"disease": "Squamous cell carcinoma", 
            "solution": "Seek medical advice for surgical removal, radiation, or topical treatments.",
            "link": "https://www.cdc.gov/squamous_cell_carcinoma"},
        7: {"disease": "Tinea Ringworm Candidiasis", 
            "solution": "Antifungal medications such as creams or oral treatments. Maintain proper hygiene.",
            "link": "https://www.cdc.gov/fungal/diseases/ringworm/index.html"},
        8: {"disease": "Vascular lesion", 
            "solution": "Most are harmless and do not require treatment. Laser therapy is an option for cosmetic reasons.",
            "link": "https://www.aad.org/public/diseases/vascular-lesions"}
    }
    
    prediction_index = np.argmax(prediction)
    result = mapping[prediction_index]
    
    return result["disease"], result["solution"], result["link"]

# Function to preprocess the uploaded image
def your_preprocessing_function(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Route to render the homepage
@app.route('/')
def indexes():
    return render_template('indexes.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part")
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Make a prediction using the loaded model
        try:
            image = your_preprocessing_function(file_path)
            prediction = model.predict(image)
            disease, solution, link = decode_prediction(prediction)

            print(f"Prediction: {disease}, Solution: {solution}, Link: {link}")

            return render_template('indexes.html', prediction=disease, solution=solution, link=link)
        except Exception as e:
            print(e)
            print(f"Error making prediction: {e}")

    return redirect(url_for('indexes'))

if __name__ == '__main__':
    app.run(debug=True,port=5000,host="0.0.0.0")
