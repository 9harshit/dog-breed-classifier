from flask import Flask, redirect, url_for, render_template, request, session
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

@app.route("/", methods = ["POST", "GET"])  # this sets the route to this page
def home():
    if request.method == "POST":
        if request.files:
            test_image =  request.files["image"]
            test_image.save(test_image.filename)  
            test_image = image.load_img(test_image.filename, target_size = (64, 64))
            model = load_model('dog_breeder_resent50v2_10.h5')
            
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)

            index = np.argmax(result[0])

            breed_label = {0: 'beagle', 1 : 'chihuahua' , 2 : 'doberman',  3: 'french_bulldog',  4 : 'golden_retriever',  5: 'malamute',  6: 'pug',  7:'saint_bernard', 8:'scottish_deerhound',  9:'tibetan_mastiff'}

            session["result"] = breed_label[index]

        if session["result"]:
            return render_template("home.html", result = session["result"])

        else:
            print("File not found")
            return render_template("home.html", result = "Please Upload Image")

        
    else:
        return render_template("home.html", result = "PLease Upload Image")



if __name__ == "__main__":
    app.run(debug = True)
