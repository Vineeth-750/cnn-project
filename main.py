
from flask import Flask,render_template,request
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
from tensorflow.keras.preprocessing import image
app =Flask(__name__)
model = load_model('C:/Users/vineeth/Desktop/pythonProject/astro_data_model.h5')
@app.route('/')
def index():
    return render_template("index.html")
@app.route("/prediction",methods=["post"])
def prediction():
    img = request.files['img']
    img.save("img.jpg")
    new_img = image.load_img(img, target_size=(69, 69))
    new_x = image.img_to_array(new_img)
    new_x = np.expand_dims(new_x, axis=0)
    new_x = new_x / 255.0  # Normalize
    new_prediction = model.predict(new_x)
    predicted_class = np.argmax(new_prediction)
    # Map the predicted class to the corresponding label
    predicted_label = features[predicted_class]

    return "welcome to prediction"

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0')

