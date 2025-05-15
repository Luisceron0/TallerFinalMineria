"""
Script para entrenar un modelo KNeighborsClassifier usando el dataset Iris 
y guardarlo con joblib para su posterior uso.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Cargar el dataset de Iris
print("Cargando el dataset de Iris...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(url, header=None, names=column_names)

print("Forma del dataset:", iris_df.shape)
print("Primeras 5 filas del dataset:")
print(iris_df.head())

# Preparar los datos para el entrenamiento
X = iris_df.iloc[:, :-1].values  # Características (sepal_length, sepal_width, petal_length, petal_width)
y = iris_df.iloc[:, -1].values   # Etiqueta (class)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# Crear y entrenar el modelo KNeighborsClassifier
print("Entrenando el modelo KNeighborsClassifier...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluar el rendimiento del modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(conf_matrix)
print("Reporte de clasificación:")
print(class_report)

# Guardar el modelo entrenado usando joblib
print("Guardando el modelo...")
model_filename = 'knn_model.pkl'
joblib.dump(knn, model_filename)
print(f"Modelo guardado como {model_filename}")

# Guardar también el mapeo de clases para referencia posterior
class_mapping = {i: label for i, label in enumerate(np.unique(y))}
print("Mapeo de clases:", class_mapping)
joblib.dump(class_mapping, 'class_mapping.pkl')
print("Mapeo de clases guardado como 'class_mapping.pkl'")

print("Proceso completado correctamente.")
