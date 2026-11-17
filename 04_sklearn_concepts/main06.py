#비지도학습 (군집화, 차원축소)
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler

##군집화
#=== K means ===

# iris = load_iris()
# X = iris.data
#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.fit_transform(X)
#
# m = KMeans(n_clusters=3, n_init=10, random_state=42)
# m.fit(X_scaled)
# y_pred = m.predict(X_scaled)
#
# print(y_pred)
# print(m.cluster_centers_.shape)
#
# ##응집도 확인
# print("응집도: ", m.inertia_)



# === K Means + elbow ===
# iris = load_iris()
# X = iris.data
#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.fit_transform(X)
#
# ##응집도, 실루엣
# for k in range(2,20):
#     m = KMeans(n_clusters=k, n_init=10, random_state=42)
#     m.fit(X_scaled)
#     y_pred = m.predict(X_scaled)
#
#     sil_score = silhouette_score(X_scaled,y_pred)
#     print(f"[{k}]응집도: {m.inertia_}, 실루엣: {sil_score}")





##차원축소  (+주성분 분석 <PCA>)
'''
기존 것을 그대로 사용해서 정리하고 결정하는 게 아니라
주성분은 완전 새로운 features가 만들어짐
'''

##=== PCA ===
# # iris = load_iris()
# # X = iris.data
# X = load_iris().data
#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
#
# m = PCA(n_components=2)     #2는 2차원으로 줄이겠다 <predict 안하고 전처리 과정임>
# m.fit(X_scaled)
# X_2d = m.transform(X_scaled)
#
# print(X.shape)
# print(X_2d.shape)



##=== PCA + K-Means ===
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

X = load_iris().data

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)  #트랜스폼은 95->1 로 기준점을 잡아주기 위해서

model_km = KMeans(n_clusters=3, n_init=10, random_state=42)
model_km.fit(X_scaled)
labels = model_km.predict(X_scaled)

model_pca = PCA(n_components=2)
model_pca.fit(X_scaled)
X_2d = model_pca.transform(X_scaled)

print(X_2d.shape)

plt.figure(figsize=(8,6))
X_data = X_2d[:,0]
y_data = X_2d[:,1]
plt.scatter(X_data, y_data, c=labels, s=30, cmap="viridis")
plt.tight_layout()
plt.show()
'''
c는 hue 같은 역할을 하고
그래프에서 3가지 색상만 나오는 이유는 clusters=3 이라서
'''

