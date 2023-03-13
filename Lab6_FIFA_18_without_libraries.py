import csv
import random
import math

# Cargar los datos desde el archivo CSV
def load_csv(filename):
    dataset = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convertir las columnas de strings a números
def convert_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


# Dividir los datos en conjuntos de entrenamiento, validación y prueba
def split_dataset(dataset, train_percent, validation_percent):
    train_size = int(len(dataset) * train_percent)
    validation_size = int(len(dataset) * validation_percent)
    test_size = len(dataset) - train_size - validation_size
    train_set = []
    validation_set = []
    test_set = []
    dataset_copy = dataset[:]
    random.shuffle(dataset_copy)
    for i in range(train_size):
        train_set.append(dataset_copy[i])
    for i in range(train_size, train_size + validation_size):
        validation_set.append(dataset_copy[i])
    for i in range(train_size + validation_size, len(dataset_copy)):
        test_set.append(dataset_copy[i])
    return train_set, validation_set, test_set


# Calcular la precisión del modelo
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))


# Clase para construir el árbol de decisión
class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    # Calcular la entropía de un conjunto de datos
    def entropy(self, data):
        counts = {}
        for row in data:
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        entropy = 0
        for label in counts:
            probability = counts[label] / float(len(data))
            entropy -= probability * math.log(probability, 2)
        return entropy

    # Dividir los datos en dos conjuntos según el valor de una columna
    def split_data(self, data, column, value):
        left = []
        right = []
        for row in data:
            if row[column] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Encontrar el mejor corte para dividir los datos
    def find_best_split(self, data):
        best_entropy = float("inf")
        best_column = None
        best_value = None
        for column in range(len(data[0]) - 1):
            values = set([row[column] for row in data])
            for value in values:
                left, right = self.split_data(data, column, value)
                if len(left) == 0 or len(right) == 0:
                    continue
                entropy = (len(left) / len(data)) * self.entropy(left) + (
                    len(right) / len(data)
                ) * self.entropy(right)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_column = column
                    best_value = value
        return best_column, best_value

        # Construir el árbol de decisión recursivamente

    def build_tree(self, data, depth=0):
        if depth >= self.max_depth:
            return max(
                set([row[-1] for row in data]), key=[row[-1] for row in data].count
            )
        if len(set([row[-1] for row in data])) == 1:
            return data[0][-1]
        column, value = self.find_best_split(data)
        left, right = self.split_data(data, column, value)
        node = {"column": column, "value": value}
        node["left"] = self.build_tree(left, depth + 1)
        node["right"] = self.build_tree(right, depth + 1)
        return node

    # Hacer una predicción utilizando el árbol de decisión
    def predict(self, row, tree):
        if isinstance(tree, str):
            return tree
        if row[tree["column"]] < tree["value"]:
            return self.predict(row, tree["left"])
        else:
            return self.predict(row, tree["right"])


dataset = load_csv('cleaned_data.csv')

train_set, validation_set, test_set = split_dataset(dataset, 0.8, 0.1)

tree = DecisionTree()
tree_model = tree.build_tree(train_set)
actual = [row[-1] for row in validation_set]
predicted = [tree.predict(row, tree_model) for row in validation_set]
precision = accuracy(actual, predicted)
print(
    "Precisión del modelo en el conjunto de validación: {:.2f}%".format(precision * 100)
)
actual = [row[-1] for row in test_set]
predicted = [tree.predict(row, tree_model) for row in test_set]
precision = accuracy(actual, predicted)
print("Precisión del modelo en el conjunto de prueba: {:.2f}%".format(precision * 100))
