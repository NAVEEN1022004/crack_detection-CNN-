from flask import Flask, render_template, request, redirect
import os
import numpy as np
import keras
from tensorflow.keras.preprocessing import image
import pickle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

app = Flask(__name__)

# Load the trained model using pickle

model = pickle.load(open('models/cracks(CNN)_model.sav', 'rb'))

# Define function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
      
        if file.filename == '':
            return redirect(request.url)
        if file:
           
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
         
            img_array = preprocess_image(file_path)
           
            prediction = model.predict(img_array)
            predicted_class = "Positive" if prediction[0] > 0.5 else "Negative"
            
            return render_template('result.html', filename=filename, predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
