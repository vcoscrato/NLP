from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    result = request.form

    # Prepare the feature vector for prediction

    prediction = str(result['texto'])




    #pkl_file = open('classifier.pkl', 'rb')
    #classifier = pickle.load(pkl_file)
    #prediction = classifier.predict(new_vector)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run()