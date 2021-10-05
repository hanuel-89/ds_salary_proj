import flask
from flask import Flask, jsonify, request
import json
from data_input import data_in
import numpy as np
import pickle


def load_model():
    filename = "models/model.pkl"
    model = pickle.load(open(filename, 'rb'))
    return model


app = Flask(__name__)
@app.route('/predict', methods=['GET'])


def predict():
    #response = json.dumps({'response': 'yahhhhh!'})
    # Stub input featuress
    request_json = request.get_json()
    x = request_json['input']
    #print(x)
    x_in = np.array(x).reshape(1,-1)
    # Load model
    model = load_model()
    prediction = model.predict(x_in)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == "__main__":
    application.run(debug=True)