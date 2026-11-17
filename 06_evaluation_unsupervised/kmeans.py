#비지도학습 (군집화, 차원축소)
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
# print("응집도: ", m.inertia_)



#=== KMeans + elbow ===
iris = load_iris()
X = iris.data

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.fit_transform(X)

for k in range(2, 10):
    m = KMeans(n_clusters=k, random_state=42, n_init=10)
    m.fit(X_scaled)
    y_pred = m.predict(X_scaled)

    sil_score = silhouette_score(X_scaled, y_pred)
    print(f"[{k}]응집도: {m.inertia_}, 실루엣: {sil_score}")

'''
#인사이트
[2]응집도: 222.36170496502308, 실루엣: 0.5817500491982808
[3]응집도: 139.82049635974982, 실루엣: 0.45994823920518635
[4]응집도: 114.09254690403088, 실루엣: 0.38694104154427816

-> 응집도 2가 가장 이상적이다
    도메인 지식이 있다면 3
'''


