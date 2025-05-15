"""
Servidor web Flask que carga un modelo KNeighborsClassifier previamente entrenado
y proporciona un endpoint para hacer predicciones en tiempo real.
"""
from flask import Flask, request, jsonify
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

@app.route('/')
def home():
    return """
    <h1>Servidor de Predicción de Flores Iris</h1>
    <p>Este servidor utiliza un modelo KNeighborsClassifier para predecir la especie de flores Iris.</p>
    <h2>Cómo usar el API:</h2>
    <p>Enviar una solicitud POST al endpoint /predict con los siguientes parámetros:</p>
    <ul>
        <li><strong>sepal_length</strong>: Longitud del sépalo (cm)</li>
        <li><strong>sepal_width</strong>: Ancho del sépalo (cm)</li>
        <li><strong>petal_length</strong>: Longitud del pétalo (cm)</li>
        <li><strong>petal_width</strong>: Ancho del pétalo (cm)</li>
    </ul>
    <p>Ejemplo de solicitud JSON:</p>
    <pre>
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    </pre>
    """

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
    app.run(host='0.0.0.0', port=port, debug=True)
