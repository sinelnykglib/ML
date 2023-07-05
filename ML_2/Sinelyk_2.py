import random

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import *
data = pd.read_csv('KM-03-3.csv')

#2.
# Виводимо кількість об'єктів кожного класу
class_counts = data['GT'].value_counts()
print('Кількість об\'єктів кожного класу:')
print(class_counts)

# Визначаємо збалансованість набору даних
balance = class_counts[1] / class_counts[0]
if balance >= 0.99 and balance <= 1.01:
    print('Набір даних збалансований')
else:
    print('Набір даних не збалансований')

#3.a
y_true = data['GT']
y_pred_1 = data['Model_1_0']
y_pred_2 = data['Model_2_1']
# створення списку з різними значеннями порогу класифікатора
thresholds = [0.05, 0.25, 0.45, 0.65, 0.95]

# обчислення метрик для кожної моделі при кожному значенні порогу
for threshold in thresholds:
    # обчислення бінарних передбачень з використанням поточного порогу
    y_pred_1_binary = [0 if pred >= threshold else 1 for pred in y_pred_1]
    y_pred_2_binary = [1 if pred >= threshold else 0 for pred in y_pred_2]


# створення словника з назвами метрик та їх значеннями для кожної моделі
model_metrics = {'Model 1': [], 'Model 2': []}
data2=data.copy()
for threshold in thresholds:
    # обчислення бінарних передбачень з використанням поточного порогу
    y_pred_1_binary = [0 if pred >= threshold else 1 for pred in y_pred_1]
    y_pred_2_binary = [1 if pred >= threshold else 0 for pred in y_pred_2]

    # обчислення метрик для кожної моделі та їх збереження в словник
    for model, y_pred_binary in zip(['Model 1', 'Model 2'], [y_pred_1_binary, y_pred_2_binary]):
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        bal_acc = balanced_accuracy_score(y_true, y_pred_binary)
        youden_j = recall + bal_acc - 1
        auc_prc = roc_auc_score(y_true, y_pred_binary, average='weighted', multi_class='ovr')
        auc_roc = roc_auc_score(y_true, y_pred_binary)

        model_metrics[model].append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'bal_acc': bal_acc,
            'youden_j': youden_j,
            'auc_prc': auc_prc,
            'auc_roc': auc_roc
        })

# побудова графіків для кожної метрики та кожної моделі
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'bal_acc', 'youden_j', 'auc_prc', 'auc_roc']
for metric in metrics_names:
    #plt.figure(figsize=(8, 6))
    for model, data2 in model_metrics.items():
        x_values = [d['threshold'] for d in data2]
        y_values = [d[metric] for d in data2]
        max_value = max(y_values)
        max_value_idx = y_values.index(max_value)
        plt.plot(x_values, y_values, label=f'{model} (max={max_value:.4f})', marker='o')
        plt.annotate(f'{max_value:.4f}', (x_values[max_value_idx], max_value), ha='center', va='bottom')
    plt.xlabel('величина  порогу')
    plt.ylabel(metric.capitalize())
    plt.legend(metrics_names)
plt.show()

# визначаємо параметр К на основі дати народження
dob = '16-09'
K = int(dob.split('-')[1]) % 4
print(K)
class_1 = data[data['GT'] == 1]

percent_to_remove = (50 + 10*K)
print(percent_to_remove)
num_to_remove = round(len(class_1) * percent_to_remove / 100)
indices_to_remove = random.sample(class_1.index.tolist(), num_to_remove)

# створюємо новий набір даних без відібраних об'єктів
new_data = data.drop(indices_to_remove)

# зберігаємо новий набір даних в csv файл
print(f"відсоток  видалених  об’єктів  класу  1: {round(num_to_remove/(len(data)/100))}%")


class_counts = new_data['GT'].value_counts()
print('Кількість об\'єктів кожного класу:')
print(class_counts)

y_true = new_data['GT']
y_pred_1 = new_data['Model_1_0']
y_pred_2 = new_data['Model_2_1']

for threshold in thresholds:
    # обчислення бінарних передбачень з використанням поточного порогу
    y_pred_1_binary = [0 if pred >= threshold else 1 for pred in y_pred_1]
    y_pred_2_binary = [1 if pred >= threshold else 0 for pred in y_pred_2]


# створення словника з назвами метрик та їх значеннями для кожної моделі
model_metrics = {'Model 1': [], 'Model 2': []}
for threshold in thresholds:
    # обчислення бінарних передбачень з використанням поточного порогу
    y_pred_1_binary = [0 if pred >= threshold else 1 for pred in y_pred_1]
    y_pred_2_binary = [1 if pred >= threshold else 0 for pred in y_pred_2]

    # обчислення метрик для кожної моделі та їх збереження в словник
    for model, y_pred_binary in zip(['Model 1', 'Model 2'], [y_pred_1_binary, y_pred_2_binary]):
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        bal_acc = balanced_accuracy_score(y_true, y_pred_binary)
        youden_j = recall + bal_acc - 1
        auc_prc = roc_auc_score(y_true, y_pred_binary, average='weighted', multi_class='ovr')
        auc_roc = roc_auc_score(y_true, y_pred_binary)

        model_metrics[model].append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'bal_acc': bal_acc,
            'youden_j': youden_j,
            'auc_prc': auc_prc,
            'auc_roc': auc_roc
        })

# побудова графіків для кожної метрики та кожної моделі
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'bal_acc', 'youden_j', 'auc_prc', 'auc_roc']
for metric in metrics_names:
    for model, new_data in model_metrics.items():
        x_values = [d['threshold'] for d in new_data]
        y_values = [d[metric] for d in new_data]
        max_value = max(y_values)
        max_value_idx = y_values.index(max_value)
        plt.plot(x_values, y_values, label=f'{model} (max={max_value:.4f})', marker='o')
        plt.annotate(f'{max_value:.4f}', (x_values[max_value_idx], max_value), ha='center', va='bottom')
    plt.xlabel('величина  порогу')
    plt.ylabel(metric.capitalize())
    plt.legend(metrics_names)
plt.show()
