import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score


def dun(n_claster):
    kmeans = KMeans(n_clusters=n_claster, init='random')
    kmeans.fit(df)
    labels = kmeans.labels_
    davies_bouldin_index = davies_bouldin_score(df, labels)
    return davies_bouldin_index


df = pd.read_csv("dataset3_l5.csv", sep=";")

# 1 task
print(df.head())

# 2 task
print('\nКількість записів у data frame:', df.shape[1])

# 3 task
df = df.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
print(df.dtypes)
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].str.replace(',', '.').astype(float)
    else:
        df[column] = df[column].replace(',', '.').astype(float)
print(df)

# 4 task
print('\nАтрибути набору даних:\n', df.columns)

# 5 task
#-------------elbow method----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
sse = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=47, init='random')
    kmeans.fit(df)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(2, 11), sse, curve="convex", direction="decreasing")
optimal_k = kl.elbow

plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('SSE (сума квадратів відхилень)')
plt.title('Elbow Method')
plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.show()

print("\nelbow method")
print("Оптимальна кількість кластерів:", optimal_k)

# -------------average silhouette method----------------
silhouette_scores = []
for n_clusters in range(2, 11):
    clusterer = KMeans(n_clusters=n_clusters, init='random', random_state=47)
    cluster_labels = clusterer.fit_predict(df)
    silhouette_avg = silhouette_score(df, cluster_labels)
    silhouette_scores.append([n_clusters, silhouette_avg])

y = [sublist[1] for sublist in silhouette_scores]
plt.plot(range(2, 11), y)
plt.xlabel('n_clusters')
plt.ylabel('average Silhouette Score')
plt.title('average silhouette method')
plt.show()

best_cluster = max(silhouette_scores, key=lambda x: x[1])
print("\naverage silhouette method")
print("Оптимальна кількість кластерів average silhouette method =", best_cluster[0])

#-------------prediction  strength  method --------------
prediction_strengths = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=47).fit(df)
    labels = kmeans.labels_
    distance_matrix = pairwise_distances(df)
    intra_distances = np.array(
        [np.mean(distance_matrix[labels == i][:, labels == i]) for i in range(n_clusters)])
    inter_distances = np.array(
        [np.mean(distance_matrix[labels == i][:, labels != i]) for i in range(n_clusters)])
    prediction_strength = np.mean(intra_distances) / np.mean(inter_distances)
    prediction_strengths.append([n_clusters, prediction_strength])

y = [sublist[1] for sublist in prediction_strengths]
plt.plot(range(2, 11), y)
plt.xlabel('n_clusters')
plt.ylabel('prediction strength')
plt.title('prediction  strength  method')
plt.show()

optimal_clusters = max(prediction_strengths, key=lambda x: x[1])
print("\nprediction  strength  method")
print(f'Оптимальна кількість кластерів: {optimal_clusters[0]}')


print('Результати отримали різні - 5, 10, 2')
print("Davies-Bouldin index для першого методу:", dun(5))
print("Davies-Bouldin index для другого методу:", dun(10))
print("Davies-Bouldin index для третього методу:", dun(2))
print('Отримали, що значення індексу Девіса-Боулдіна найменше для verage silhouette method(найменше розсіяні) тому обираємо 10')

optimal_n_cluster = 10
kmeans = KMeans(n_clusters=optimal_n_cluster, init='random').fit(df)
cluster_centers = kmeans.cluster_centers_
print("Координати центрів оптимального кластера(10):")
for center in cluster_centers:
    print(center)

#task 6
best_silhouette_score = -1
best_wcss = float('inf')
best_kmeans_model = None

for k in range(10):
    kmeans = KMeans(n_clusters=10, init='k-means++')
    kmeans.fit(df)
    labels = kmeans.labels_
    silhouette = silhouette_score(df, labels)
    wcss = kmeans.inertia_

    if silhouette > best_silhouette_score and wcss < best_wcss:
        best_silhouette_score = silhouette
        best_wcss = wcss
        best_kmeans_model = kmeans

print("Найкраща кластеризація:")
print("Кількість кластерів:", best_kmeans_model.n_clusters)
print("Коефіцієнт силуету:", best_silhouette_score)
print("WCSS:", best_wcss)

print('Критерії для відбору: коефіцієнт силуету (silhouette coefficient) та внутрішня сума квадратів відстаней (WCSS)')

print("Координати найкращих центрів:")
centroids = best_kmeans_model.cluster_centers_
for centroid in centroids:
    print(centroid)

#task 7
clustering = AgglomerativeClustering(n_clusters=10)
clustering.fit(df)
cluster_labels = clustering.labels_
aglomerative = silhouette_score(df, cluster_labels)

cluster_centers = []
for cluster_label in range(10):
    cluster_data = df[cluster_labels == cluster_label]
    cluster_center = cluster_data.mean()
    cluster_centers.append(cluster_center)
print("AgglomerativeClustering координати центрів кластера:")
for cluster_center in cluster_centers:
    print(cluster_center)

# task 8
x = ['k-means++', 'Agglomerative']
y = [best_silhouette_score, aglomerative]
plt.bar(x, y)
plt.show()
print('k-середніх показав краще значення (ближче до 1) коефіцієнт силуету (Silhouette Coefficient)')
