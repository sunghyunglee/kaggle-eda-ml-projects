#자원축소 (주성분 분석 <PCA>)
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

##PCA
# X = load_iris().data
#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#
# m = PCA(n_components=2)
# m.fit(X_scaled)
# X_2d = m.transform(X_scaled)
#
# print(X.shape)
# print(X_2d.shape)



##PCA + KMeans

X = load_iris().data

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

model_km = KMeans(n_clusters=3, random_state=42, n_init=10)
model_km.fit(X_scaled)
labels = model_km.predict(X_scaled)

model_pca = PCA(n_components=2)
model_pca.fit(X_scaled)
X_2d = model_pca.transform(X_scaled)

print(X_2d.shape)

plt.figure(figsize = (8,6))
X_data = X_2d[:,0]
y_data = X_2d[:,1]
plt.scatter(X_data,y_data, c=labels, s=30, cmap="viridis")
plt.tight_layout()
plt.show()


