#DL <분류형>
##breast_cancer
'''
Epoch 50: Loss(CossEntropy) 0.2280
Epoch 100: Loss(CossEntropy) 0.0807
Epoch 150: Loss(CossEntropy) 0.0521
Epoch 200: Loss(CossEntropy) 0.0367
DL Accuracy:  0.9824561403508771
'''

import sklearn
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#데이터 준비
X,y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)   #분류는 long 타입
y_test = torch.tensor(y_test, dtype=torch.long)


#모델 설계
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),   #출력 2개 (클래스 개수만큼)
)


#손실함수 & 옵티마이저
loss_fn = nn.CrossEntropyLoss()     #교차엔트로피 (분류전용)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#학습 루프 (경사하강법 + 역전파는 여기안에 있음)
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)     #순전파
    loss = loss_fn(output, y_train)     #교차엔트로피로 손실 계산
    loss.backward()     #역전파: 기울기 계산
    optimizer.step()    #경사하강법: 가중치 업데이트

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss(CossEntropy) {loss.item():.4f}")


#평가
model.eval()
with torch.no_grad():
    output = model(X_test)
    y_pred = torch.argmax(output, dim=1)    #2개 출력 중 더 큰쪽을 정답으로 선택

acc = accuracy_score(y_test, y_pred)
print("DL Accuracy: ", acc)

