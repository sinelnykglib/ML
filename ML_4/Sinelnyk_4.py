import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


#1
dt = pd.read_csv('dataset3_l4.csv')

cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
dict = {}
for i in cols:
    dt[i], dict[i] = pd.factorize(dt[i])

#2
num_records = dt.shape[0]
print("Кількість записів: ", num_records)

#3
columns = dt.columns
print("Атрибути набору даних: ", columns)

#4
num_splits = int(input("Введіть кількість варіантів перемішування (не менше 3): "))

features = dt.iloc[:, :-1].values
labels = dt.iloc[:, -1].values

# Ініціалізація генератора випадкових чисел
random_state = 42

# Створення об’єкту ShuffleSplit та отримання навчальної та тестової вибірок
ss = ShuffleSplit(n_splits=num_splits, test_size=0.2, random_state=random_state)
for train_index, test_index in ss.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Виведення розмірів навчальної та тестової вибірок на екран
print("Розмір навчальної вибірки:", X_train.shape[0])
print("Розмір тестової вибірки:", X_test.shape[0])

# Перевірка збалансованості набору даних
counts = pd.value_counts(labels)
print("Розподіл класів у наборі даних:\n", counts)

#5

X = dt.drop("NObeyesdad", axis=1)
y = dt["NObeyesdad"]

# Розділити вибірку на тренувальну та тестову
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Створити та навчити модель
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Оцінити якість класифікації на тестовій вибірці
accuracy = knn.score(X_test, y_test)
print("Accuracy:", accuracy)

#6
train_acc = knn.score(X_train, y_train)
train_pred = knn.predict(X_train)
train_cm = confusion_matrix(y_train, train_pred)
train_cr = classification_report(y_train, train_pred)

# обчислення метрик для тестової вибірки
test_acc = knn.score(X_test, y_test)
test_pred = knn.predict(X_test)
test_cm = confusion_matrix(y_test, test_pred)
test_cr = classification_report(y_test, test_pred)

# виведення результатів
print("Training Accuracy:", train_acc)
print("Training Confusion Matrix:")
print(train_cm)
print("Training Classification Report:")
print(train_cr)

print("Testing Accuracy:", test_acc)
print("Testing Confusion Matrix:")
print(test_cm)
print("Testing Classification Report:")
print(test_cr)

# відображення теплової карти матриці помилок для тестової вибірки
plt.figure(figsize=(8,6))
sns.heatmap(test_cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#7
train_scores = []
test_scores = []

for p in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=5, p=p)
    knn.fit(X_train, y_train)
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    train_scores.append([accuracy_score(y_train, train_pred), precision_score(y_train, train_pred, average='macro'),
                         recall_score(y_train, train_pred, average='macro'), f1_score(y_train, train_pred, average='macro')])
    test_scores.append([accuracy_score(y_test, test_pred), precision_score(y_test, test_pred, average='macro'),
                        recall_score(y_test, test_pred, average='macro'), f1_score(y_test, test_pred, average='macro')])

train_scores = np.array(train_scores)
test_scores = np.array(test_scores)

# plot the results
plt.plot(range(1, 21), train_scores[:, 0], label='Train Accuracy')
plt.plot(range(1, 21), test_scores[:, 0], label='Test Accuracy')
plt.xlabel('p')
plt.ylabel('Accuracy')
plt.title('Effect of Minkowski distance parameter on accuracy')
plt.legend()
plt.show()

plt.plot(range(1, 21), train_scores[:, 3], label='Train F1-score')
plt.plot(range(1, 21), test_scores[:, 3], label='Test F1-score')
plt.xlabel('p')
plt.ylabel('F1-score')
plt.title('Effect of Minkowski distance parameter on F1-score')
plt.legend()
plt.show()