import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
CLFmodel = pickle.load(open('model.pkl', 'rb'))  # load the ML model

# The route() decorator to tell Flask what URL should trigger our function.
# ‘/’ is the root of the website, such as www.westga.edu
@flask_app.route("/")   
def index():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])   
def predict():
    credit = request.form.get("credit")
    area = request.form.get("area")
    results = [[]]
    if (credit is None or area is None):
        return render_template("index.html", predicted_text = "Selection Not Made")
    if credit == "good":
        if area == "rural":
            prediction = CLFmodel.predict([[1, 1, 0, 0]])
        if area == "urban":
            prediction = CLFmodel.predict([[1, 0, 1, 0]])
        else:
            prediction = CLFmodel.predict([[1, 0, 0, 1]])
    else:
        if area == "rural":
            prediction = CLFmodel.predict([[0, 1, 0, 0]])
        if area == "urban":
            prediction = CLFmodel.predict([[0, 0, 1, 0]])
        else:
            prediction = CLFmodel.predict([[0, 0, 0, 1]])
    if (prediction == [1]):
        prediction = "Approved"
    else:
        prediction = "Not Approved"        
    return render_template("index.html", predicted_text = prediction)       
if __name__ =="__main__":
    flask_app.run(debug = True)

                                            
