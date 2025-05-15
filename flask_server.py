"""
Servidor web Flask que carga un modelo KNeighborsClassifier previamente entrenado
y proporciona un endpoint para hacer predicciones en tiempo real con interfaz web.
"""
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import os

app = Flask(__name__)

# Cargar el modelo entrenado y el mapeo de clases
print("Cargando el modelo KNN...")
model = joblib.load('knn_model.pkl')
try:
    class_mapping = joblib.load('class_mapping.pkl')
    print("Mapeo de clases cargado:", class_mapping)
except:
    print("No se encontró un mapeo de clases. Las predicciones serán numéricas.")
    class_mapping = None

# HTML Template para el frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Flores Iris</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e9f7ef;
            border-radius: 5px;
            border-left: 5px solid #2ecc71;
        }
        .result h2 {
            margin-top: 0;
            color: #27ae60;
        }
        .api-info {
            margin-top: 40px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        pre {
            background-color: #f1f1f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .error {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        .iris-images {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        .iris-image {
            text-align: center;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: white;
        }
        .iris-image img {
            max-width: 150px;
            height: auto;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predicción de Especies de Iris</h1>
        
        <div class="iris-images">
            <div class="iris-image">
                <div style="width:150px;height:150px;background-color:#f0f8ff;display:flex;align-items:center;justify-content:center;">
                    <span>Iris Setosa</span>
                </div>
                <p>Iris Setosa</p>
            </div>
            <div class="iris-image">
                <div style="width:150px;height:150px;background-color:#e6e6fa;display:flex;align-items:center;justify-content:center;">
                    <span>Iris Versicolor</span>
                </div>
                <p>Iris Versicolor</p>
            </div>
            <div class="iris-image">
                <div style="width:150px;height:150px;background-color:#e0ffff;display:flex;align-items:center;justify-content:center;">
                    <span>Iris Virginica</span>
                </div>
                <p>Iris Virginica</p>
            </div>
        </div>
        
        <form id="predictionForm" method="post" action="/predict_web">
            <div class="form-group">
                <label for="sepal_length">Longitud del Sépalo (cm):</label>
                <input type="number" step="0.1" id="sepal_length" name="sepal_length" value="5.1" min="0" max="10" required>
            </div>
            <div class="form-group">
                <label for="sepal_width">Ancho del Sépalo (cm):</label>
                <input type="number" step="0.1" id="sepal_width" name="sepal_width" value="3.5" min="0" max="10" required>
            </div>
            <div class="form-group">
                <label for="petal_length">Longitud del Pétalo (cm):</label>
                <input type="number" step="0.1" id="petal_length" name="petal_length" value="1.4" min="0" max="10" required>
            </div>
            <div class="form-group">
                <label for="petal_width">Ancho del Pétalo (cm):</label>
                <input type="number" step="0.1" id="petal_width" name="petal_width" value="0.2" min="0" max="10" required>
            </div>
            <button type="submit">Predecir Especie</button>
        </form>
        
        {% if prediction %}
        <div class="result">
            <h2>Resultado de la Predicción</h2>
            <p><strong>Especie predicha:</strong> {{ prediction_label }}</p>
            <p><strong>Características de entrada:</strong></p>
            <ul>
                <li>Longitud del Sépalo: {{ features.sepal_length }} cm</li>
                <li>Ancho del Sépalo: {{ features.sepal_width }} cm</li>
                <li>Longitud del Pétalo: {{ features.petal_length }} cm</li>
                <li>Ancho del Pétalo: {{ features.petal_width }} cm</li>
            </ul>
        </div>
        {% endif %}
        
        {% if error %}
        <div class="error">
            <p>{{ error }}</p>
        </div>
        {% endif %}
        
        <div class="api-info">
            <h2>Información del API</h2>
            <p>Este servidor también proporciona un endpoint API para realizar predicciones programáticamente.</p>
            <h3>Endpoint: /predict</h3>
            <p>Enviar una solicitud POST con los siguientes parámetros en formato JSON:</p>
            <pre>
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
            </pre>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_web', methods=['POST'])
def predict_web():
    try:
        # Obtener los datos del formulario
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
        
        # Preparar los datos para la predicción
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Realizar la predicción
        prediction = model.predict(features)[0]
        
        # Si existe un mapeo de clases, convertir la predicción numérica a la etiqueta correspondiente
        if class_mapping:
            prediction_label = class_mapping.get(np.where(model.classes_ == prediction)[0][0], prediction)
        else:
            prediction_label = prediction
            
        # Devolver la plantilla con el resultado
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=prediction,
            prediction_label=prediction_label,
            features={
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        )
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de la solicitud
        data = request.get_json(force=True)
        
        # Verificar que los datos contienen las características esperadas
        required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for feature in required_features:
            if feature not in data:
                return jsonify({'error': f'Falta la característica {feature}'}), 400
        
        # Preparar los datos para la predicción
        features = np.array([
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]).reshape(1, -1)
        
        # Realizar la predicción
        prediction = model.predict(features)[0]
        
        # Si existe un mapeo de clases, convertir la predicción numérica a la etiqueta correspondiente
        if class_mapping:
            prediction_label = class_mapping.get(np.where(model.classes_ == prediction)[0][0], prediction)
        else:
            prediction_label = prediction
            
        # Devolver la predicción
        return jsonify({
            'prediction': prediction,
            'prediction_label': prediction_label,
            'input_features': {
                'sepal_length': data['sepal_length'],
                'sepal_width': data['sepal_width'],
                'petal_length': data['petal_length'],
                'petal_width': data['petal_width']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Comprobar si los archivos del modelo existen
    if not os.path.exists('knn_model.pkl'):
        print("ERROR: El archivo 'knn_model.pkl' no existe. Ejecute primero train_model.py")
        exit(1)
    
    # Iniciar el servidor
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor en el puerto {port}...")
    print(f"Accede a la interfaz web en http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)