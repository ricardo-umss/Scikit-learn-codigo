import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs

# Generar datos sintéticos para ejemplo - 300 puntos de datos, 3 centros,
data, true_labels = make_blobs(n_samples=300, centers=3, random_state=12)

# Crear listas para almacenar los datos- se obtiene las horas y calificaciones de la variable data
horas_estudio = data[:, 0]
calificaciones = data[:, 1]
kmeans_clusters = []
dbscan_clusters = []

# Aplicar el algoritmo K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_clusters = kmeans.fit_predict(data)

# Aplicar el algoritmo DBSCAN epsilon distancia minima 0.5 - minimo de puntos 5 para formar cluster
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_clusters = dbscan.fit_predict(data)

#Calculo de métricas

# Calcular puntuación de Silueta para K-Means
silueta_kmeans = silhouette_score(data, kmeans_clusters)
# Calcular puntuación de Silueta para DBSCAN
silueta_dbscan = silhouette_score(data, dbscan_clusters)
# Calcular el índice de Calinski-Harabasz para K-Means
# que separados están los clústeres y qué tan compactos son dentro de ellos
calinski_kmeans = calinski_harabasz_score(data, kmeans_clusters)
# Calcular el índice de Calinski-Harabasz para DBSCAN
calinski_dbscan = calinski_harabasz_score(data, dbscan_clusters)
# Calcular el índice Davies-Bouldin para K-Means
davies_kmeans = davies_bouldin_score(data, kmeans_clusters)
# Calcular el índice Davies-Bouldin para DBSCAN
davies_dbscan = davies_bouldin_score(data, dbscan_clusters)


# Visualización de los resultados
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.scatter(horas_estudio, calificaciones, c=kmeans_clusters, cmap='viridis')
plt.title("Agrupamiento con K-Means")


plt.subplot(122)
plt.scatter(horas_estudio, calificaciones, c=dbscan_clusters, cmap='viridis')
plt.title("Agrupamiento con DBSCAN")

plt.show()

# Visualización de las métricas en gráficos separados
plt.figure(figsize=(12, 4))

# Gráfico de Silhouette Score
plt.subplot(131)
plt.bar(["K-Means", "DBSCAN"], [silueta_kmeans, silueta_dbscan])
plt.xlabel("Algoritmo")
plt.ylabel("Puntuación de Silueta")
plt.title("Puntuación de Silueta")

# Gráfico de Calinski-Harabasz Score
plt.subplot(132)
plt.bar(["K-Means", "DBSCAN"], [calinski_kmeans, calinski_dbscan])
plt.xlabel("Algoritmo")
plt.ylabel("Índice de Calinski-Harabasz")
plt.title("Índice de Calinski-Harabasz")

# Gráfico de Davies-Bouldin Score
plt.subplot(133)
plt.bar(["K-Means", "DBSCAN"], [davies_kmeans, davies_dbscan])
plt.xlabel("Algoritmo")
plt.ylabel("Índice Davies-Bouldin")
plt.title("Índice Davies-Bouldin")

plt.tight_layout()
plt.show()

# Resultados por consola
print(f"Silueta qué tan similar es un objeto a su propio clúster en comparación con otros clústeres cercanos.")
print(f"Puntuación de Silueta para K-Means: {silueta_kmeans}")
print(f"Puntuación de Silueta para DBSCAN: {silueta_dbscan}")
print(f"Calinski qué tan separados están los clústeres y qué tan compactos son dentro de ellos")
print(f"Índice de Calinski-Harabasz para K-Means: {calinski_kmeans}")
print(f"Davies evalúa la compacidad y la separación de los clústeres")
print(f"Índice Davies-Bouldin para K-Means: {davies_kmeans}")
print(f"Índice Davies-Bouldin para DBSCAN: {davies_dbscan}")
