import pandas as pd
from random import seed
from random import randrange
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv('high_diamond_ranked_10min.csv')
df.info()
df.head()
df.describe()
df.corr()

X = df[['blueKills', 'blueTowersDestroyed', 'blueTotalGold', 'blueTotalMinionsKilled']].values
y = df['blueWins'].values

# Dividir los datos en conjunto de entrenamiento y prueba
def train_test_split(dataset, split=0.8):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Fijar la semilla aleatoria para reproducibilidad
seed(42)

# Dividir los datos en entrenamiento y prueba
train_data, test_data = train_test_split(list(zip(X, y)))
X_train = [x[0] for x in train_data]
y_train = [x[1] for x in train_data]
X_test = [x[0] for x in test_data]
y_test = [x[1] for x in test_data]

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs, self.tree) for inputs in X]

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Si no hay más ejemplos, devolver un nodo con la etiqueta más común
        if n_samples == 0:
            return {'label': self._majority_label(y)}

        # Si todos los ejemplos pertenecen a la misma clase, devolver un nodo con esa etiqueta
        if n_labels == 1:
            return {'label': y[0]}

        # Si se alcanza la profundidad máxima, devolver un nodo con la etiqueta más común
        if depth >= self.max_depth:
            return {'label': self._majority_label(y)}

        # Si no se cumplen las condiciones anteriores, continuar con la división
        feature_idxs = np.random.choice(n_features, min(n_features, int(np.ceil(np.sqrt(n_features)))), replace=False)
        best_feature_idx, best_threshold = self._best_split(X, y, feature_idxs)

        # Si no se encontró una división que mejore la ganancia de información, devolver un nodo con la etiqueta más común
        if best_feature_idx is None:
            return {'label': self._majority_label(y)}

        # Si se cumplen las condiciones para hacer una división, continuar con la recursión
        left_idxs = X[:, best_feature_idx] < best_threshold
        right_idxs = X[:, best_feature_idx] >= best_threshold
        left_tree = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right_tree = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)

        # Devolver un nodo que represente la división realizada
        return {'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left': left_tree,
                'right': right_tree}


def _predict(self, inputs, tree):
    # Si se alcanza una hoja, devolver la etiqueta correspondiente
    if 'label' in tree:
        return tree['label']

    # Si no se alcanza una hoja, continuar con la recursión en el subárbol correspondiente
    if inputs[tree['feature_idx']] < tree['threshold']:
        return self._predict(inputs, tree['left'])
    else:
        return self._predict(inputs, tree['right'])

def _best_split(self, X, y, feature_idxs):
    best_gain = -1
    best_feature_idx = None
    best_threshold = None
    n_samples, n_features = X.shape

    for i in feature_idxs:
        thresholds = np.unique(X[:, i])
        for threshold in thresholds:
            gain = self._information_gain(y, X[:, i], threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = i
                best_threshold = threshold

    if best_feature_idx is not None:
        return best_feature_idx, best_threshold
    else:
        return None, None

def _information_gain(self, y, feature, threshold):
    # Calcular la entropía antes de la división
    parent_entropy = self._entropy(y)

    # Calcular la entropía después de la división
    left_idxs = feature < threshold
    right_idxs = feature >= threshold
    left_entropy = self._entropy(y[left_idxs])
    right_entropy = self._entropy(y[right_idxs])
    child_entropy = (sum(left_idxs) / len(feature)) * left_entropy + (sum(right_idxs) / len(feature)) * right_entropy

    # Devolver la ganancia de información
    return parent_entropy - child_entropy

def _entropy(self, y):
    n_samples = len(y)
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n_samples
    return sum(-p * np.log2(p) for p in probs if p > 0)

def _majority_label(self, y):
    _, counts = np.unique(y, return_counts=True)
    return max(zip(counts, range(len(counts))))[1]

# Instanciar un modelo de Árboles de Decisión
tree = DecisionTree(max_depth=10, min_samples_split=2)

# Ajustar el modelo con los datos de entrenamiento
tree.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = tree.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Precisión del modelo: {accuracy}')

