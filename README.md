# Taller Final de Minería de Datos

Este proyecto implementa un modelo de machine learning (KNeighborsClassifier) entrenado con el conjunto de datos Iris y expone un servidor web con Flask para realizar predicciones en tiempo real.

## Estructura del Proyecto

- `train_model.py`: Script para entrenar y guardar el modelo KNN.
- `app.py`: Servidor web Flask que proporciona un endpoint para predicciones.
- `test_server.py`: Script para probar el servidor de predicciones.
- `requirements.txt`: Archivo con las dependencias del proyecto.

## Requisitos Previos

Asegúrate de tener instalado Python 3.7 o superior. Se recomienda usar un entorno virtual para la instalación.

## Instalación

1. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   ```

2. Activa el entorno virtual:
   - En Windows:
     ```bash
     venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Paso 1: Entrenar el Modelo

Ejecuta el script para entrenar el modelo KNeighborsClassifier con el conjunto de datos Iris:

```bash
python train_model.py
```

Esto generará dos archivos:
- `knn_model.pkl`: El modelo entrenado.
- `class_mapping.pkl`: El mapeo de clases numéricas a nombres de flores.

### Paso 2: Iniciar el Servidor Flask

Una vez entrenado el modelo, inicia el servidor Flask:

```bash
python app.py
```

El servidor estará disponible en `http://localhost:5000/`.

### Paso 3: Realizar Predicciones

Puedes realizar predicciones de dos formas:

1. Usando el script de prueba:
   ```bash
   python test_server.py
   ```

2. Enviando solicitudes POST directamente al endpoint `/predict`:

   Ejemplo usando curl:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' http://localhost:5000/predict
   ```

   Ejemplo usando Postman:
   - Método: POST
   - URL: http://localhost:5000/predict
   - Headers: Content-Type: application/json
   - Body (raw, JSON):
     ```json
     {
         "sepal_length": 5.1,
         "sepal_width": 3.5,
         "petal_length": 1.4,
         "petal_width": 0.2
     }
     ```

## Estructura de la Respuesta

El servidor responderá con un JSON con la siguiente estructura:

```json
{
    "prediction": "Iris-setosa",
    "prediction_label": "Iris-setosa",
    "input_features": {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
}
```

## Notas

- El modelo está entrenado para clasificar flores Iris en tres especies: setosa, versicolor y virginica.
- El servidor Flask está configurado para ejecutarse en modo debug. Para un entorno de producción, desactiva esta opción.
