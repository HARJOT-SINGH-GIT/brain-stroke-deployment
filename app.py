from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
with open('Brain_Stroke.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = [
                int(request.form['gender']),
                float(request.form['age']),
                int(request.form['hypertension']),
                int(request.form['heart_disease']),
                int(request.form['ever_married']),
                int(request.form['work_type']),
                int(request.form['residence_type']),
                float(request.form['avg_glucose_level']),
                float(request.form['bmi']),
                int(request.form['smoking_status'])
            ]

            # Convert the input data to a numpy array and reshape for the model
            input_array = np.array([input_data])

            # Predict using the model
            prediction = model.predict(input_array)

            # Convert the prediction to a JSON response
            return jsonify({'prediction': prediction.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return "Invalid request method", 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
