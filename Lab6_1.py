import csv
import random
import math

# Cargar los datos y separar la variable objetivo de las features
data = []
with open('high_diamond_ranked_10min.csv') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        data.append(row)

features = ['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled', 'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
target = 'blueWins'

# Dividir los datos en conjunto de entrenamiento, validación y prueba
random.shuffle(data)
train_size = int(0.8 * len(data))
val_size = test_size = int(0.1 * len(data))
train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

# Función para contar las etiquetas (0 o 1) en un conjunto de datos
def count_labels(data):
    counts = [0, 0]
    for row in data:
        if len(row) > 0:
            counts[int(row[0])] += 1
    return counts

# Función para calcular la precisión del modelo en un conjunto de datos
def accuracy(data, tree):
    correct = 0
    for row in data:
        prediction = classify(row, tree)
        if prediction == int(row[0]):
            correct += 1
    return correct / len(data)

# Función para construir el árbol de decisión
def build_tree(data, depth):
    # Si todos los ejemplos tienen la misma etiqueta, devolver un nodo hoja
    labels = [int(row[0]) for row in data]
    if len(set(labels)) == 1:
        return {'label': labels[0]}

    # Si se llega a la profundidad máxima, devolver un nodo hoja con la etiqueta más común
    if depth == 0:
        return {'label': max(set(labels), key=labels.count)}

    # Encontrar la mejor feature y punto de división
    best_feature = None
    best_value = None
    best_gain = 0
    for i in range(1, len(data[0])):
        values = set([float(row[i]) for row in data])
        for value in values:
            gain = information_gain(data, i, value)
            if gain > best_gain:
                best_feature = i
                best_value = value
                best_gain = gain

    # Dividir el conjunto de datos en dos subconjuntos y construir el árbol recursivamente
    left_data = [row for row in data if float(row[best_feature]) <= best_value]
    right_data = [row for row in data if float(row[best_feature]) > best_value]
    left_tree = build_tree(left_data, depth-1)
    right_tree = build_tree(right_data, depth-1)

    # Devolver un nodo interno con la feature de división y los subárboles izquierdo y derecho
    return {'feature': headers[best_feature], 'value': best_value, 'left': left_tree, 'right': right_tree}

def classify(row, tree):
    if 'label' in tree:
        return tree['label']
    else:
        if float(row[headers.index(tree['feature'])]) <= tree['value']:
            return classify(row, tree['left'])
        else:
            return classify(row, tree['right'])
        
def entropy(data):
    counts = count_labels(data)
    proportions = [count / len(data) for count in counts]
    return -sum(p * math.log2(p) for p in proportions if p > 0)

def information_gain(data, feature_index, value):
    left_data = [row for row in data if float(row[feature_index]) <= value]
    right_data = [row for row in data if float(row[feature_index]) > value]
    left_entropy = entropy(left_data)
    right_entropy = entropy(right_data)
    return entropy(data) - (len(left_data) / len(data)) * left_entropy - (len(right_data) / len(data)) * right_entropy

tree = build_tree(train_data, depth=5)

best_tree = None
best_accuracy = 0
for depth in range(1, 11):
    tree = build_tree(train_data, depth=depth)
    acc = accuracy(val_data, tree)
    if acc > best_accuracy:
        best_tree = tree
        best_accuracy = acc

accuracy = accuracy(test_data, best_tree)
print('Accuracy: %.2f' % accuracy)





