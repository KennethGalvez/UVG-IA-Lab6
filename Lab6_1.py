import csv
import random
import math

# Paso 1: Importar y leer el dataset
filename = 'high_diamond_ranked_10min.csv'
dataset = []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    headers = next(csvreader)
    for row in csvreader:
        dataset.append(row)

# Paso 2: Dividir el dataset en conjunto de entrenamiento, validación y prueba
def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio[0])
    valid_size = int(len(dataset) * split_ratio[1])
    train_set = []
    valid_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    while len(valid_set) < valid_size:
        index = random.randrange(len(test_set))
        valid_set.append(test_set.pop(index))
    return [train_set, valid_set, test_set]

split_ratio = [0.8, 0.1, 0.1]
train_set, valid_set, test_set = split_dataset(dataset, split_ratio)

# Paso 3: Definir las características y la variable objetivo del modelo
features = ['blueWins', 'blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled', 'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
target = 'blueWins'

# Paso 4: Entrenar el modelo con el conjunto de entrenamiento
def calc_entropy(dataset, features, target):
    class_counts = {}
    for row in dataset:
        if row[features.index(target)] not in class_counts:
            class_counts[row[features.index(target)]] = 0
        class_counts[row[features.index(target)]] += 1
    entropy = 0
    for count in class_counts.values():
        prob = count / float(len(dataset))
        entropy += -prob * math.log(prob, 2)
    return entropy

def mode(lst):
    counter = {}
    for elem in lst:
        if elem not in counter:
            counter[elem] = 0
        counter[elem] += 1
    max_count = 0
    modes = []
    for k,v in counter.items():
        if v > max_count:
            max_count = v
            modes = [k]
        elif v == max_count:
            modes.append(k)
    return modes[0]

# Paso 4: Entrenar el modelo con el conjunto de entrenamiento
def build_tree(dataset, features, target):
    class_values = list(set(row[features.index(target)] for row in dataset))
    if len(class_values) == 1:
        return class_values[0]
    best_feature = None
    best_gain = 0
    for feature in features:
        if feature == target:
            continue
        feature_values = list(set(row[features.index(feature)] for row in dataset))
        entropy = 0
        for value in feature_values:
            subset = [row for row in dataset if row[features.index(feature)] == value]
            prob = len(subset) / float(len(dataset))
            entropy += prob * calc_entropy(subset, features, target)
        info_gain = calc_entropy(dataset, features, target) - entropy
        if info_gain > best_gain:
            best_feature = feature
            best_gain = info_gain
    if best_feature == None:
        return mode([row[features.index(target)] for row in dataset])
    tree = {best_feature:{}}
    feature_values = list(set(row[features.index(best_feature)] for row in dataset))
    for value in feature_values:
        subset = [row for row in dataset if row[features.index(best_feature)] == value]
        subtree = build_tree(subset, [f for f in features if f != best_feature], target)
        tree[best_feature][value] = subtree
    return tree

tree = build_tree(train_set, features, target)

# Paso 5: Evaluar el modelo con el conjunto de validación
def predict(tree, row):
    if type(tree) == str:
        return tree
    else:
        feature = list(tree.keys())[0]
        if row[features.index(feature)] not in tree[feature]:
            return mode([row[features.index(target)] for row in train_set])
        subtree = tree[feature][row[features.index(feature)]]
        return predict(subtree, row)

def evaluate_accuracy(tree, dataset):
    correct = 0
    for row in dataset:
        prediction = predict(tree, row)
        if prediction == row[features.index(target)]:
            correct += 1
    return correct / float(len(dataset))

accuracy = evaluate_accuracy(tree, valid_set)
print("Accuracy on validation set:", accuracy)

# Paso 6: Ajustar los hiperparámetros del modelo y volver a entrenar y evaluar
max_depths = [5, 10, 15, 20, 25]
best_accuracy = 0
best_tree = None
for max_depth in max_depths:
    features = ['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled', 'redKills', 'redTowersDestroyed', 'redTotalGold', 'redTotalMinionsKilled']
    target = 'blueWins'
    tree = build_tree(train_set, features, target)
    accuracy = evaluate_accuracy(tree, valid_set)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_tree = tree

print("Best accuracy on validation set:", best_accuracy)

# Paso 7: Evaluar el modelo con el conjunto de prueba
accuracy = evaluate_accuracy(best_tree, test_set)
print("Accuracy on test set:", accuracy)


