from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    bedrooms = float(request.form['bedrooms'])
    bathrooms = float(request.form['bathrooms'])

    prediction = model.predict([[sqft, bedrooms, bathrooms]])

    return render_template('index.html', prediction_text="Price: {}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)