from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

# Load the saved model and scaler
model = tf.keras.models.load_model('bodyfat_model.h5')
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

app = Flask(__name__)

# Default initial values
default_values = {
    'Density': '1.0708',
    'Age': '23',
    'Weight': '70',  # in kg
    'Height': '172',  # in inches
    'Neck': '36.2',
    'Chest': '93.1',
    'Abdomen': '85.2',
    'Hip': '94.5',
    'Thigh': '59.0',
    'Knee': '37.3',
    'Ankle': '21.9',
    'Biceps': '32.0',
    'Forearm': '27.4',
    'Wrist': '17.1'
}

def preprocess_input(data):
    data = np.array(data).reshape(1, -1)
    data = (data - scaler_mean) / scaler_scale  # Scale input
    return data

def convert_to_model_units(data):
    """
    Convert the data from the client to the units expected by the model.
    - Convert weight from kilograms to pounds.
    - Convert height from inches to centimeters.
    """
    weight_in_kg = data[2]  # Weight is the third element in the list
    height_in_cm = data[3]  # Height is the fourth element in the list

    # Conversion formulas
    weight_in_pounds = weight_in_kg / 2.20462
    height_in_inches = height_in_cm / 2.54

    # Replace the original values in the data
    data[2] = weight_in_pounds
    data[3] = height_in_inches

    return data

@app.route('/', methods=['GET', 'POST'])
def home():
    global default_values
    if request.method == 'POST':
        # Save form values to global variable
        old_values = {field: request.form[field] for field in request.form}
        return render_template('index.html', old_values=old_values)
    return render_template('index.html', old_values=default_values)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Lấy dữ liệu đầu vào từ form
            data = [float(request.form[field]) for field in request.form]
            print(f"Original data: {data}")
            
            # Chuyển đổi sang các đơn vị mà mô hình yêu cầu
            data = convert_to_model_units(data)
            print(f"Converted data: {data}")
            
            # Tiền xử lý dữ liệu đầu vào
            processed_data = preprocess_input(data)
            
            # Dự đoán
            prediction = model.predict(processed_data)[0][0]
            
            # Chuyển đổi prediction từ float32 sang float (Python float)
            prediction = float(prediction)
            
            # Trả về giá trị dự đoán dưới dạng JSON
            return jsonify({'prediction': round(prediction, 2)})
        except Exception as e:
            return jsonify({'error': str(e)})
    return jsonify({'error': 'Invalid request method'})



if __name__ == '__main__':
    app.run(debug=True)
