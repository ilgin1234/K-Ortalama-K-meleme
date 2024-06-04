import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df2 = pd.read_csv("C:/Users/Lenovo/Downloads/breast+cancer+wisconsin+original/wdbc.data")

file_path=("C:/Users/Lenovo/Downloads/breast+cancer+wisconsin+original/breast-cancer-wisconsin.data")
#------------------------------------clustering-----------------------------------------

column_names = ['ID', 'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape', 'Marginal_adhesion',
                'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
df = pd.read_csv(file_path, header=None, names=column_names)

# 'Bare_nuclei' sütunundaki '?' değerlerini NaN ile değiştirme ve eksik verileri doldurma
df['Bare_nuclei'] = pd.to_numeric(df['Bare_nuclei'], errors='coerce')
df.fillna(df.mean(), inplace=True)

# ID ve Class sütunları dışındaki sütunları seçme
X = df.iloc[:, 1:-1].values
y = df['Class'].values

# Veriyi ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA ile özellikleri iki boyuta indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Ortalama Kümeleme modelini oluşturma ve eğitme
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_pca)

# Küme merkezlerini ve etiketleri alma
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# İki boyutlu veriyi kullanarak görselleştirme
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('PCA ile İki Boyuta İndirgenmiş K-Ortalama Kümeleme')
plt.show()

#------------------------------------------------------------------------------------------
"""
# Veri setinin sütun adlarını ayarlama
df2.columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]

# 'Diagnosis' sütununu sayısallaştırma (M = 1, B = 0)
df2['Diagnosis'] = df2['Diagnosis'].map({'M': 1, 'B': 0})

# Kullanılacak özellikleri seçme (ID ve Diagnosis sütunlarını hariç tutarak tüm sayısal özellikler)
X = df2.iloc[:, 2:].values  # Feature_1'den Feature_30'a kadar olan sütunlar

# K-Ortalama Kümeleme modelini oluşturma ve eğitme
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Küme merkezlerini ve etiketleri alma
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Kümeleme sonuçlarını görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-Ortalama Kümeleme')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.show()
"""

