from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

#definisikan flask
app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])

def weight_prediction():
    if request.method == 'GET':
        return render_template("predict_web.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_features = dict(request.form).values()   #proses request 'from' atau 'dari' user input
        #Linear Regression
        weight_features = np.array([float(x) for x in weight_features]).reshape(1,-1)
        model = joblib.load("model-development/whg_liReg.pkl")
        print(weight_features)
        result = model.predict(weight_features)
        return render_template('predict_web.html', result=result)
    else:
        return "Unsupported Request Method"

if __name__ == '__main__':
    app.run(port=5000, debug = True)
