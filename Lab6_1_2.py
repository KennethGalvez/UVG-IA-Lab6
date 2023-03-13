import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Cargue los datos en un dataframe
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Divida los datos en conjuntos de entrenamiento, validación y prueba
train_val, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

# Seleccione las características y la variable objetivo
features = ['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled',
            'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
target = 'blueWins'

# Separe las características y la variable objetivo en conjuntos de entrenamiento, validación y prueba
X_train = train[features]
y_train = train[target]
X_val = val[features]
y_val = val[target]
X_test = test[features]
y_test = test[target]

# Cree un modelo de Árbol de Decisión utilizando el conjunto de entrenamiento
model = DecisionTreeClassifier(random_state=42)

# Ajuste los parámetros del modelo utilizando el conjunto de validación
model.fit(X_train, y_train)

# Mida la precisión del modelo utilizando el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Grafique el árbol de decisión para visualizar cómo se toman las decisiones en el modelo
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=features, class_names=["Red Wins", "Blue Wins"])
plt.show()

# Realice predicciones con el modelo y mida la precisión de las predicciones utilizando el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión de las predicciones:", accuracy)

