#DL <회귀형>
##california_housing
'''
Epoch 50: Loss(MSE) 0.4353
Epoch 100: Loss(MSE) 0.3726
Epoch 150: Loss(MSE) 0.3440
Epoch 200: Loss(MSE) 0.3066
DL MAEL:  0.39812472462654114
DL RMSE:  0.5684205117395033
DL R2:  0.7534343004226685
'''

import torch
import torch.nn as nn
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

#데이터 준비
X,y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# torch tensor로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


#모델 설계 (뉴런 쌓기)
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)


#손실 함수 & 옵티마이저
loss_fn = nn.MSELoss()  #손실 함수: 얼마나 틀렸는지 측정
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   #옵티마이저: 오차 줄이기


#학습 루프
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()   #이전 기울기 초기화
    y_pred = model(X_train)     #순전파: 예측
    loss = loss_fn(y_pred, y_train)     #손실 계산
    loss.backward()     #역전파: 기울기 계산
    optimizer.step()    #가충치 업데이트

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss(MSE) {loss.item():.4f}")


#평가
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("DL MAEL: ", mae)
print("DL RMSE: ", rmse)
print("DL R2: ", r2)

