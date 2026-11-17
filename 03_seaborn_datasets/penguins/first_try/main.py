#구조 -> 결측치 -> 분포 (그래프) -> 관계- > 모델

#penguins

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

#데이터
df = sns.load_dataset("penguins")

# print(df.head())
# print(df.describe())
# print(df.info())


#결측치, 중복제거, 인코딩
# print(df.isna().sum())
# print(df.isna().mean())
##결측치 적을땐 삭제 하지만 index 부조화가 일어나서 -> .reset_index()
df = df.dropna().reset_index(drop=True)


#단변량 분석
##y 값 분석
print(df["species"].value_counts())
##x 값 분석
##1.x 숫자형 (부리길이, 부리깊이, 몸무게, 지느러미길이)
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#
# sns.histplot(df["bill_length_mm"], bins=20, ax=axes[0, 0])
# axes[0, 0].set_title("부리 길이 분포", fontweight="bold")
#
# sns.histplot(df["bill_depth_mm"], bins=20, ax=axes[0, 1])
# axes[0, 1].set_title("부리 깊이 분포", fontweight="bold")
#
# sns.histplot(df["body_mass_g"], bins=20, ax=axes[1, 0])
# axes[1, 0].set_title("몸무게 분포", fontweight="bold")
#
# sns.histplot(df["flipper_length_mm"], bins=20, ax=axes[1, 1])
# axes[1, 1].set_title("지느러미길이 분포", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##2.x 범주형 (종, 섬, 성별)
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))
#
# sns.countplot(data=df, x="species", ax=axes[0])
# axes[0].set_title("종", fontweight="bold")
#
# sns.countplot(data=df, x="island", ax=axes[1])
# axes[1].set_title("섬", fontweight="bold")
#
# sns.countplot(data=df, x="sex", ax=axes[2])
# axes[2].set_title("성별", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#이변량 분석
##종별 신체 치수 평균
# print(df.groupby("species")["bill_length_mm"].mean())
# print(df.groupby("species")["bill_depth_mm"].mean())
# print(df.groupby("species")["flipper_length_mm"].mean())
# print(df.groupby("species")["body_mass_g"].mean())
#
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# sns.boxplot(data=df, ax=axes[0, 0], x="species", y="bill_length_mm")
# axes[0, 0].set_title("종별 / 부리 길이 분포", fontweight="bold")
# sns.boxplot(data=df, ax=axes[0, 1], x="species", y="bill_depth_mm")
# axes[0, 1].set_title("종별 / 부리 깊이 분포", fontweight="bold")
# sns.boxplot(data=df, ax=axes[1, 0], x="species", y="flipper_length_mm")
# axes[1, 0].set_title("종별 / 지느러미 길이 분포", fontweight="bold")
# sns.boxplot(data=df, ax=axes[1, 1], x="species", y="body_mass_g")
# axes[1, 1].set_title("종별 / 몸무게 분포", fontweight="bold")
#
# plt.tight_layout()
# plt.show()

##종별 / 섬 관계, 성별 / 몸무게 관계
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
# sns.barplot(data=df, ax=axes[0], x="species", y="island") #이상하게 나옴
# result = pd.crosstab(df["species"], df["island"])
# print(result)
#
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# sns.heatmap(result, annot=True, ax=axes[0], fmt="d")
# axes[0].set_title("종별 / 섬", fontweight="bold")
#
# sns.barplot(data=df, x="species", y="body_mass_g", ax=axes[1], hue="sex") #hue -> 성별 구분을 위해
# axes[1].set_title("성별 / 몸무게", fontweight="bold")
#
# plt.tight_layout()
# plt.show()


#상관관계
fig, axes = plt.subplots(1,2, figsize=(10,8))
df_nums = df[["bill_length_mm", "bill_depth_mm", "body_mass_g", "flipper_length_mm"]]
cor = df_nums.corr()
sns.heatmap(cor, annot=True, ax=axes[0], cmap="coolwarm", fmt=".2f")
axes[0].set_title("신체 치수 상관관계", fontweight="bold")

##산점도: 부리길이, 부리깊이
sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=axes[1])
axes[1].set_title("부리길이 / 부리깊이 상관관계", fontweight="bold")

plt.tight_layout()
plt.show()


#인사이트
'''
##종별 몸무게 파악 (종(species) → 몸무게(body_mass_g))
젠투펭귄(Gentoo)은 평균 5092.4g으로
다른 두 종(species)보다 몸무게(body_mass_g)가 1300g 이상 압도적으로 무겁습니다.
    - (아델리펭귄(Adelie)은 3706.2g, 턱끈펭귄(Chinstrap)은 3733.1g 이다.)

##섬별 몸무게 파악 (섬(island) → 몸무게(body_mass_g)) <교란변수 주의>
섬(island)별 몸무게 차이는 착시에 가깝습니다.
    - 토르거슨섬(Torgersen)엔 아델리펭귄만 있고, 드림섬(Dream)엔 아델리(55)+턱끈(68)이 섞여 있다.
    - 비스코섬(Biscoe)은 젠투(119)뿐 아니라 아델리(44)도 함께 산다 
        → "비스코=젠투 전용"은 아니지만, 반대로 젠투는 비스코에만 100% 산다.
    - 즉 "섬 효과"가 아니라 "그 섬에 어떤 종 비율로 분포하는가"가 실질적 원인이며, 
    특히 젠투→비스코 방향은 거의 1:1 대응이라 island를 피처로 쓰면 species 정보가 사실상 새어 들어가는 셈이다.

##체급 구간에 따른 지느러미 길이 파악
body_mass_g를 4분위(WeightBand)로 나누면
188.7mm → 192.9mm → 203.6mm → 219.6mm 로,
체급 구간(WeightBand)이 커질수록 지느러미 길이(flipper_length_mm)도 함께, 그리고 거의 일정한 간격으로 길어진다.

##성별 x 종별 몸무게 파악
모든 종에서 수컷(male)이 암컷(female)보다 무겁고, 그 차이는 젠투펭귄(약 805g)에서 가장 크다.
    - (아델리 약 675g, 턱끈 약 412g 차이)

##상관관계 파악
지느러미 길이는 몸무게(body_mass_g)와 강한 양의 상관(0.87)이고
부리 깊이(bill_depth_mm)는 음의 상관(-0.47)을 보입니다.
    - 즉, 몸이 클수록 날개(지느러미)도 크며
    - 부리 깊이는 개체 크기가 아니라 종 특성(젠투만 유독 얕음: 15.0 vs 나머지 18점대)을 반영하기 때문에 
    역방향 상관이 나타난 것으로 보인다.

##부리 길이 x 부리 깊이 산점도 파악
아델리(38.82)와 턱끈(48.83)은 부리 깊이(18점대)로는 거의 안 갈리지만
부리 길이 하나만으로 두 종이 뚜렷이 나뉘고,
젠투는 부리 깊이(15.0)가 유독 낮아 산점도에서 아예 다른 축에 떨어져 보인다.
'''

