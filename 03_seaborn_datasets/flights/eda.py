#flights

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


#데이터
df = sns.load_dataset("flights")

print(df.head())
print(df.tail())
print(df.info())
# print(df.describe())
## -> year, month, passengers


#결측치, 중복제거, 인코딩
print(df.isna().sum())
print(df.isnull().mean())
# -> 결측치 0


#단변량 분석
##y값 분석 (passengers)
print(df["passengers"].value_counts())
##x값 분석 (year, month)
###x 범주형
# -> countplot / boxplot / barplot 해봤지만 다 이상하게 나와서 바로 이변량으로 넘어감

#날짜로 바꾸기
x = df["year"].astype(str) + "-" + df["month"].astype(str)
df["date"] = pd.to_datetime(x, format="%Y-%b")
df = df.sort_values("date").reset_index(drop=True)


##이변량 분석 년도별/승객, 월별/승객
# print(df.groupby("passengers")["year"].mean())
# print(df.groupby("passengers")["month"].mean())   #둘다 에러남
print(df.groupby("year")["passengers"].mean())
print(df.groupby("month")["passengers"].mean())

# fig, axes = plt.subplots(1,2, figsize=(15,10))
#
# sns.barplot(data=df, x="month", y="passengers", ax=axes[0])
# axes[0].set_title("년도별/승객", fontweight="bold")
#
# sns.barplot(data=df, x="year", y="passengers", ax=axes[1])
# axes[1].set_title("월별/승객", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계
##year, month, passengers
df["month_num"] = df["date"].dt.month
df_nums = df[["year", "month_num", "passengers"]]
cor = df_nums.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("상관관계", fontweight="bold")

plt.tight_layout()
plt.show()


#인사이트
'''
1. 추세(Trend) — 우상향
    연도가 지날수록 승객 수가 꾸준히 증가합니다. 
    그냥 증가가 아니라 점점 가속화되는(지수형에 가까운) 증가라는 게 핵심 포인트

2. 계절성(Seasonality) — 반복 패턴
    매년 여름(7~8월)에 승객 수가 튀어 오르고, 
    겨울(11~1월)에 상대적으로 적은 연 단위 반복 패턴

3. 추세와 계절성이 곱셈적(multiplicative)으로 결합
    초반(1949년)엔 여름-겨울 차이가 크지 않다가, 후반(1960년)으로 갈수록 계절 진폭 자체가 커집니다. 
    이게 시계열 분석에서 가장 중요한 관찰 포인트
    
    -> X = pd.concat([df["t"], month_dummies], axis=1)
       y = np.log(df["passengers"])   # <-- 핵심: 타겟을 로그 변환
       #예측 (로그 스케일)
       y_pred_log = m.predict(X_test)
       #원래 스케일로 복원
       y_pred = np.exp(y_pred_log)
       y_test_original = np.exp(y_test)   # 또는 그냥 df["passengers"]의 해당 구간 사용
'''

