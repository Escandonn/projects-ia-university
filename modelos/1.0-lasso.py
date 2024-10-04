import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Cargar el archivo Excel
ruta_archivo = 'housing.csv'
data = pd.read_csv(ruta_archivo)

# Ver las primeras filas del dataset para confirmar que se ha cargado correctamente
print(data.head())

# Preparar los datos
X = data.drop(['Price', 'Address'], axis=1)  # características
y = data['Price']  # precio de la casa

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Regresión Lasso
lasso_model = Lasso(alpha=0.1, max_iter=10000)

# Entrenar el modelo
lasso_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = lasso_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse:.2f}')

# Coeficientes del modelo
print('Coeficientes del modelo:')
print(lasso_model.coef_)

# Características más importantes
print('Características más importantes:')
print(X.columns[lasso_model.coef_!= 0])

# Performance o score del modelo
score = lasso_model.score(X_test, y_test)
print(f'Score del modelo: {score:.2f}')

# Cross-validation
scores = cross_val_score(lasso_model, X, y, cv=5)
print(f'Promedio de scores en cross-validation: {scores.mean():.2f}')

# Predicción de un ejemplo
ejemplo = X_test.iloc[0]
prediccion = lasso_model.predict(ejemplo.values.reshape(1, -1))
print(f'Predicción para el ejemplo: {prediccion[0]:.2f}')