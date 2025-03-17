from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO


app = Flask(__name__)
model = load_model('/Users/vineeth/Desktop/pythonProject/astro_data_model.h5')
img_path = 'static/uploads/img.jpg'
# Define your features list with corresponding labels
features = ['Disk,Face-on,No Spiral', 'Smooth, Completely round', 'Smooth,in-between round', 'Smooth,Cigar shaped',
            'Disk,Edge-on,Rounded Bulge', 'Disk,Edge-on,Boxy Bulge', 'Disk,Edge-on,No Bulge',
            'Disk,Face-on,Tight Spiral', 'Disk,Face-on,Medium Spiral',
            'Disk,Face_on,Loose Sprial']  # Update with your actual labels

@app.route('/')
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    img_data = request.files['img'].read()
    img = Image.open(BytesIO(img_data))

    img_array = image.img_to_array(img.resize((69, 69)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    new_prediction = model.predict(img_array)
    predicted_class = np.argmax(new_prediction)

    # Map the predicted class to the corresponding label
    predicted_label = features[predicted_class]

    # Optionally, you can save the image if needed:

    img.save(img_path)
    return render_template("detector.html",data1=predicted_class,data2=predicted_label,data3=img_path)

@app.route('/about')
def about():
    return render_template("howitworks.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
