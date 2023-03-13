import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import re
from ydata_profiling import ProfileReport


def compute_operation(cell):
    # Utiliza expresiones regulares para extraer la operación
    match = re.match(r"(\d+)\s*([+-])\s*(\d+)", str(cell))
    if match:
        # Si la celda contiene una operación, calcula el resultado y lo devuelve
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        if op == "+":
            return a + b
        else:
            return a - b
    else:
        # Si la celda no contiene una operación, devuelve el valor original
        return cell


# Cargar dataset
data = pd.read_csv("CompleteDataset.csv", low_memory=False)

# Generar un informe detallado de los datos
profile = ProfileReport(data, title="CompleteDataset", explorative=True)

# Guardar el informe en un archivo HTML
profile.to_file("CompleteDataset.html")

# Aplicar la función compute_operation a todas las columnas
data = data.applymap(compute_operation)

# Eliminar las filas con valores faltantes
data = data.dropna()

# Eliminar las columnas innecesarias
data.drop(
    [
        "ID",
        "Name",
        "Age",
        "Photo",
        "Nationality",
        "Flag",
        "Club Logo",
        "Value",
        "Wage",
        "Special",
        "Preferred Positions",
        "Number",
        "Club",
    ],
    axis=1,
    inplace=True,
)

data.to_csv("cleaned_data.csv", index=False)

# Dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba 80%, 10%, 10%
train, valtest = train_test_split(data, test_size=0.2, random_state=1)
val, test = train_test_split(valtest, test_size=0.5, random_state=1)

# Separar las características y la variable objetivo
train_x = train.drop(["Potential"], axis=1)
train_y = train["Potential"]
val_x = val.drop(["Potential"], axis=1)
val_y = val["Potential"]
test_x = test.drop(["Potential"], axis=1)
test_y = test["Potential"]

# Entrenar el modelo de árbol de decisión
model = DecisionTreeRegressor(random_state=1)
model.fit(train_x, train_y)

# Obtener las importancias de las características y ordenarlas en orden descendente
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Mostrar las 5 características principales
print("Top 5 features:")
for f in range(5):
    print("%d. %s (%f)" %
          (f + 1, train_x.columns[indices[f]], importances[indices[f]]))

# Evaluar el modelo en el conjunto de validación
y_val_pred = model.predict(val_x)
print("R2 score on validation set: ", r2_score(val_y, y_val_pred))

# Evaluar el modelo en el conjunto de prueba
y_test_pred = model.predict(test_x)
print("R2 score on test set: ", r2_score(test_y, y_test_pred))

# Ajustar los hiperparámetros
params = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_samples_split": [2, 5, 10, 20],
}
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42), params, cv=5, n_jobs=-1
)
grid_search.fit(pd.concat([train_x, val_x]), pd.concat([train_y, val_y]))

print("Best hyperparameters: ", grid_search.best_params_)

# Predecir en el set de testing usando el mejor modelo encontrado con grid Search
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(test_x)

# Clacular el puntaje R2 en el set de testing
print("R2 score on test set using best model: ", r2_score(test_y, y_test_pred))
