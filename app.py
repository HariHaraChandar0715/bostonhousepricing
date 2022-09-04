import pickle
import json
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 


app=Flask(__name__)
### Load the model
regmodel=pickle.load(open("regmodel.pkl","rb"))
scalar=pickle.load(open("scaling.pkl","rb"))
### Home Page
@app.route('/')
def home():
    return render_template('home.html')


### Creating the predict API
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    predoutput=regmodel.predict(new_data)
    print(predoutput[0])
    return jsonify(predoutput[0]) 

if __name__=="__main__":
    app.run(port=5000,debug=True)