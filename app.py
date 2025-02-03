from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = 'models/ETR.pkl'
model = None
try:
    model = joblib.load(model_path, mmap_mode='r')
    print("Model loaded successfully with memory mapping.")
except MemoryError as e:
    print(f"Memory error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model could not be loaded due to memory error.")

    try:
        # Check if a CSV file is uploaded
        if 'csv_file' in request.files:
            csv_file = request.files['csv_file']
            if csv_file.filename != '':
                # Read the CSV file
                data = pd.read_csv(csv_file)
                # Ensure the CSV has the correct columns
                required_columns = [
                    'Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)',
                    'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)',
                    'Time at 4.15V (s)', 'Time constant current (s)',
                    'Charging time (s)', 'RUL'
                ]
                if all(column in data.columns for column in required_columns):
                    # Process the data as needed
                    inputs = data[required_columns[:-1]].values
                    # Predict RUL for each row
                    predictions = model.predict(inputs)
                    # Return the predictions
                    return render_template('result.html', rul=predictions)
                else:
                    return render_template('index.html', error="CSV file does not have the required columns.")

        # Check if a CSV line is entered
        csv_line = request.form.get('csv_line')
        if csv_line:
            try:
                # Convert the CSV line to a numpy array
                inputs = np.array([float(x) for x in csv_line.split(',')])
                if inputs.shape[0] != 9:
                    raise ValueError("CSV line does not have the required number of values.")
                inputs = inputs[1:-1].reshape(1, -1)  # Exclude the Cycle_Index and RUL values
                # Predict RUL
                rul = model.predict(inputs)[0]
                # Display the RUL estimate
                return render_template('result.html', rul=rul)
            except ValueError:
                return render_template('index.html', error="Please enter a valid CSV line with 9 numerical values.")

        # Collect input values from the form
        inputs = [
            float(request.form['F1']),  # Discharge Time (s)
            float(request.form['F2']),  # Time at 4.15V (s)
            float(request.form['F3']),  # Time Constant Current (s)
            float(request.form['F4']),  # Decrement 3.6-3.4V (s)
            float(request.form['F5']),  # Max. Voltage Discharge (V)
            float(request.form['F6']),  # Min. Voltage Charge (V)
            float(request.form['F7'])   # Charging Time (s)
        ]
        inputs = np.array(inputs).reshape(1, -1)

        # Predict RUL
        rul = model.predict(inputs)[0]

        # Display the RUL estimate
        return render_template('result.html', rul=rul)
    except ValueError:
        return render_template('index.html', error="Please enter valid numerical values.")

if __name__ == '__main__':
    app.run(debug=True)