"""
Script para probar el servidor Flask de predicción de flores Iris.
"""
import requests
import json

# URL del servidor (ajusta según sea necesario)
server_url = 'http://localhost:5000/predict'

# Datos de ejemplo para la predicción
# Estos son datos de ejemplo para diferentes especies de Iris
test_samples = [
    # Iris-setosa
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },
    # Iris-versicolor
    {
        "sepal_length": 6.0,
        "sepal_width": 2.2,
        "petal_length": 4.0,
        "petal_width": 1.0
    },
    # Iris-virginica
    {
        "sepal_length": 6.7,
        "sepal_width": 3.3,
        "petal_length": 5.7,
        "petal_width": 2.1
    }
]

def test_prediction(sample):
    """Envía una solicitud POST al servidor y muestra la respuesta."""
    try:
        # Convertir el sample a formato JSON
        headers = {"Content-Type": "application/json"}
        data_json = json.dumps(sample)
        
        # Enviar la solicitud POST
        print(f"Enviando solicitud con datos: {data_json}")
        response = requests.post(server_url, data=data_json, headers=headers)
        
        # Mostrar la respuesta
        if response.status_code == 200:
            result = response.json()
            print("\nRespuesta exitosa:")
            print(f"Predicción: {result['prediction']}")
            print(f"Etiqueta de predicción: {result['prediction_label']}")
            print(f"Características de entrada: {result['input_features']}")
            print("-" * 50)
        else:
            print(f"\nError en la solicitud (Código {response.status_code}):")
            print(response.text)
            print("-" * 50)
            
    except Exception as e:
        print(f"\nError al enviar la solicitud: {str(e)}")

# Ejecutar pruebas para cada muestra
print("Iniciando pruebas del servidor de predicción...")
for i, sample in enumerate(test_samples, 1):
    print(f"\nPrueba #{i}:")
    test_prediction(sample)

print("\nPruebas completadas.")
