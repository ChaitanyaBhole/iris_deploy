# -*- coding: utf-8 -*-
# PREDICTION API WITH HTML FORM

import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

with open('rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')  # Homepage
def home():
    return render_template('index.html')
    
    # return "<H1> Please use the <a href=http://127.0.0.1:5000/predict> /predict route </a> to access the model </H1>"

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction

    if prediction==0:
        pred_class = 'iris-verscicolor'
    elif prediction == 1:
        pred_class = 'iris-setosa'
    else:
        pred_class = 'iris-virginica'
        
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(pred_class)) # rendering the predicted result


if __name__ == '__main__':
    app.run()   
    