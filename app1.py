
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('/Users/vineeth/Desktop/pythonProject/astro_data_model.h5')

# Define your features list with corresponding labels
features = ['Disk,Face-on,No Spiral','Smooth, Completely round','Smooth,in-between round','Smooth,Cigar shaped',
            'Disk,Edge-on,Rounded Bulge',
            'Disk,Edge-on,Boxy Bulge','Disk,Edge-on,No Bulge','Disk,Face-on,Tight Spiral','Disk,Face-on,Medium Spiral',
            'Disk,Face_on,Loose Sprial']  # Update with your actual labels

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():
    img = request.files['img'].read()
    img = image.img_to_array(image.img_to_array(image.load_img(img, target_size=(69, 69))))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize

    new_prediction = model.predict(img)
    predicted_class = np.argmax(new_prediction)

    # Map the predicted class to the corresponding label
    predicted_label = features[predicted_class]

    return f"Predicted class: {predicted_class}, Predicted label: {predicted_label}"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
