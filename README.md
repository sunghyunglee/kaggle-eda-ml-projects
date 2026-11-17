# kaggle-eda-ml-project

Kaggle 대회 데이터와 seaborn 내장 데이터로 **EDA → 전처리 → 모델 학습 → 평가 → 캐글 제출**까지 직접 해 본 연습 모음입니다.
scikit-learn 모델 비교, 교차검증·혼동행렬 같은 평가 방법, 비지도학습(KMeans·PCA), 같은 데이터로 PyTorch MLP와 sklearn 비교까지 담았습니다.
대부분 PyCharm에서 직접 작성했습니다.

## 폴더 구성

| # | 폴더 | 내용 |
|---|---|---|
| 01 | [titanic](01_titanic) | 베이스라인 → 특징 추가 단계별(step1~3) → EDA(단변량·이변량·상관관계) → 복습판(review) |
| 02 | [spaceship_titanic](02_spaceship_titanic) | EDA(CryoSleep·지출·Deck/Side) + RandomForest 하이퍼파라미터 조정 |
| 03 | [seaborn_datasets](03_seaborn_datasets) | penguins · tips · flights · diamonds · mpg 각각 EDA + 모델 (처음 버전 `first_try/`, 복습 버전 `review/`, 정리판 `eda.py`·`model.py`) |
| 04 | [sklearn_concepts](04_sklearn_concepts) | 회귀(선형·다항·규제), 분류, VotingClassifier, 불균형 데이터, 트리·부스팅, KMeans·PCA, 이미지 특징 추출, 퍼셉트론, SimpleImputer + 파이프라인 |
| 05 | [model_comparison](05_model_comparison) | `make_pipeline(StandardScaler(), 모델)`로 LR · KNN · SVM · RF 4개 모델을 7개 데이터에서 비교 |
| 06 | [evaluation_unsupervised](06_evaluation_unsupervised) | K-fold 교차검증, 혼동행렬·classification_report, 결정트리·앙상블, KMeans(엘보우·실루엣), PCA 시각화, 퍼셉트론·MLP(XOR) |
| 07 | [pytorch](07_pytorch) | 텐서 기초, MLP 회귀(california), MLP 분류(breast_cancer) |

## 결과 요약

### Kaggle 제출

| 대회 | 모델 | Public Score |
|---|---|---|
| Titanic | RandomForest (Pclass, Sex, Age, Fare, FamilySize, Title) | **0.78468** |
| Spaceship Titanic | RandomForest (n_estimators=1000, max_depth=9) | **0.80196** (첫 시도 0.76455) |

Spaceship Titanic은 EDA 결과를 반영해 특징을 바꿨더니 오히려 0.80009로 떨어졌습니다. "EDA에서 보인 차이가 항상 점수로 이어지지는 않는다"는 걸 배운 사례입니다.

### seaborn 데이터

| 데이터 | 유형 | LinearRegression / LogisticRegression | RandomForest |
|---|---|---|---|
| penguins | 분류 ACC | 0.986 | 1.000 |
| tips | 회귀 R2 | **0.518** | 0.246 |
| flights | 시계열 회귀 R2 | **0.558** (시간 + 월 더미) | — |
| diamonds | 회귀 R2 | 0.867 | **0.982** |
| mpg | 회귀 R2 | 0.844 | **0.917** |

tips는 특징을 많이 넣을수록 오히려 점수가 떨어져서, 의미 있는 특징만 고르는 게 중요하다는 걸 확인했습니다.

### 4개 모델 비교 (05_model_comparison)

| 데이터 | 지표 | LR | KNN | SVM | RF |
|---|---|---|---|---|---|
| iris | ACC | 1.000 | 1.000 | 1.000 | 1.000 |
| wine | ACC | 1.000 | 0.944 | 1.000 | 1.000 |
| breast_cancer | ACC | 0.974 | 0.947 | **0.982** | 0.956 |
| digits | ACC | 0.972 | 0.975 | **0.981** | 0.972 |
| taxis (payment) | ACC | 0.941 | 0.939 | 0.935 | **0.946** |
| california_housing | R2 | 0.576 | 0.670 | 0.728 | **0.803** |
| diabetes | R2 | **0.453** | 0.425 | 0.182 | 0.435 |

### PyTorch vs scikit-learn (07_pytorch)

| 데이터 | PyTorch MLP | sklearn |
|---|---|---|
| california_housing (R2) | **0.753** | LR 0.576 / RF 0.803 |
| breast_cancer (ACC) | **0.982** | SVM 0.982 / LR 0.974 |

## 데이터

- Titanic, Spaceship Titanic: Kaggle 대회 규정상 원본 `train.csv`·`test.csv`는 올리지 않았습니다. 각 대회 페이지에서 내려받아 폴더 안 `data/`에 넣으면 실행됩니다.
  - https://www.kaggle.com/competitions/titanic
  - https://www.kaggle.com/competitions/spaceship-titanic
- seaborn 데이터는 `sns.load_dataset()`으로 불러옵니다. `first_try/data/`에 있는 csv도 seaborn 공개 데이터입니다.
- 04~07의 iris, wine, breast_cancer, digits, diabetes, california_housing은 `sklearn.datasets`로 불러옵니다.

## 실행 환경

- Python 3.12
- pandas, numpy, matplotlib, seaborn, scikit-learn, torch
- 그래프 한글 폰트: `Malgun Gothic` (Windows 기준)
