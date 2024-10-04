import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
data = pd.read_csv('diabetes.csv')

# Ver las primeras filas del dataset para confirmar que se ha cargado correctamente
print(data.head())

# Preparar los datos
X = data.drop(['Salida'], axis=1)  # características
y = data['Salida']  # variable objetivo

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Regresión Logística
logistic_model = LogisticRegression(max_iter=1000)

# Entrenar el modelo
logistic_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = logistic_model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión: {accuracy:.2f}')

# Informe de clasificación
print('Informe de clasificación:')
print(classification_report(y_test, y_pred))

# Matriz de confusión
print('Matriz de confusión:')
print(confusion_matrix(y_test, y_pred))

# Coeficientes del modelo
print('Coeficientes del modelo:')
print(logistic_model.coef_)

# Características más importantes
print('Características más importantes:')
print(X.columns[logistic_model.coef_[0]!= 0])

# Resultado de Performance o score del modelo
print('Resultado de Performance o score del modelo:')
print(logistic_model.score(X_test, y_test))

# Usando la función score del modelo
print('Usando la función score del modelo:')
print(logistic_model.score(X_train, y_train))

# Usando cross validate y generando un promedio de los resultados
scores = cross_val_score(logistic_model, X, y, cv=5)
print('Usando cross validate y generando un promedio de los resultados:')
print(f'Promedio de scores: {scores.mean():.2f}')

# Predicción de ejemplo
print('Predicción de ejemplo:')
ejemplo = X_test.iloc[0]
print(f'Entrada: {ejemplo}')
print(f'Predicción: {logistic_model.predict([ejemplo])[0]}')