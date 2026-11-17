#penguins


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터로드
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
df = pd.read_csv(url)

# 데이터파악
print(df.head())
print(df.tail())
print(df.info())
print(df.isna().sum())



# 결측치 처리
cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df[cols] = df[cols].fillna(df[cols].median())
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

# 단변량 분석
# fig, axes = plt.subplots(1, 3, figsize=(10,5))
#
# ##   - species별 countplot
# axes[0].set_title("species별 countplot", weight="bold")
# sns.countplot(x="species" , data=df, ax=axes[0])
#
# ##   - island별 countplot
# axes[1].set_title("island별 countplot", weight="bold")
# sns.countplot(x="island" , data=df, ax=axes[1])
#
# ##   - sex별 countplot
# axes[2].set_title("sex별 countplot", weight="bold")
# sns.countplot(x="sex", data=df, ax=axes[2])
#
# #   - 각 subplot에 set_title()로 제목 달기
# plt.tight_layout()
# plt.show()



# 이변량 분석 (species, island, sex, 몸무게 구간에 따른 관계 파악)
# fig, axes = plt.subplots(2,2, figsize=(15,10))
#
# ##   [0,0] species별 body_mass_g 평균 -> barplot
# print(df.groupby("species")["body_mass_g"].mean())
# axes[0,0].set_title("species별 body_mass_g 평균", weight="bold")
# sns.barplot(x="species", y="body_mass_g", data=df, ax=axes[0,0])
#
# ##   [0,1] island별 body_mass_g 평균 -> barplot
# axes[0,1].set_title("island별 body_mass_g 평균", weight="bold")
# sns.barplot(x="island", y="body_mass_g", data=df, ax=axes[0,1])
#
# ##   [1,0] feature engineering: body_mass_g를 pd.cut으로 구간화해서
# ##         WeightBand 컬럼 만들고 (bins, labels는 직접 정하기)
# ##         -> WeightBand별 flipper_length_mm 평균 barplot
# print("===== body_mass_g =====")
# print(df["body_mass_g"].describe())
# df["WeightBand"] = pd.cut(
#     df["body_mass_g"],
#     bins=[2500, 3500, 4500, 5500, 6500],
#     labels=["소형", "중소형", "중형", "대형"]
# )
# print("===== WeightBand별 flipper_length_mm 평균 =====")
# print(df.groupby("WeightBand")["flipper_length_mm"].mean())
# axes[1,0].set_title("WeightBand별 flipper_length_mm 평균", weight="bold")
# sns.barplot(x= "WeightBand", y="flipper_length_mm", data=df, ax=axes[1,0])
#
# #   [1,1] sex + species에 따른 body_mass_g 평균
# #         -> barplot에 hue="species" 활용
# #         (또는 groupby(["sex","species"])["body_mass_g"].mean().unstack() 으로 heatmap도 가능)
#
# df["SS"] = df["sex"] + df["species"]
# print("===== sex + species에 따른 body_mass_g 평균 =====")
# print(df.groupby("SS")["body_mass_g"].mean())
# axes[1,1].set_title("sex + species에 따른 body_mass_g 평균", weight="bold")
# sns.barplot(x="SS", y="body_mass_g", hue="species", data=df, ax=axes[1,1])
#
#
# #출력
# plt.tight_layout()
# plt.show()


##섬별 종 분포
# print(df.groupby("species")["body_mass_g"].mean())
# print("===== 섬별 종 분포 =====")
# plt.figure(figsize=(5, 5))
# sns.countplot(x="species", hue="island", data=df)
# plt.title("섬별 종 분포")
# plt.show()



# 상관관계
# cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
# cor = df[cols].corr()
#
# plt.figure(figsize=(10,8))
# plt.title("bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g 상관관계", weight="bold")
# sns.heatmap(cor, cmap="coolwarm", annot=True, fmt=".2f")
#
# #출력
# plt.tight_layout()
# plt.show()



# 인사이트  <핵심ver.>
'''
##종별 몸무게 파악 (종(species) → 몸무게(body_mass_g))
젠투펭귄(Gentoo)은 평균 5067.7g으로
다른 두 종(species)보다 몸무게(body_mass_g)가 1300g 이상 압도적으로 무겁습니다.
    - (아델리펭귄(Adelie)은 3703.0g, 턱끈펭귄(Chinstrap)은 3733.1g 이다.)

##섬별 몸무게 파악 (섬(island) → 몸무게(body_mass_g)) <교란변수 주의>
섬(island)별 몸무게 차이는 착시입니다.
    - 비스코섬(Biscoe)엔 젠투펭귄만, 토르거슨섬(Torgersen)엔 아델리펭귄만 산다
    - 종 분포 차이가 원인이며 "섬 효과"가 아니라 "그 섬에 어떤 종이 사는가"가 실질적 원인이다.

##체급 구간에 따른 지느러미 길이 파악
체급 구간(WeightBand)이 커질수록 지느러미 길이(flipper_length_mm)도 함께 길어진다

##성별 x 종별 몸무게 파악
모든 종에서 수컷(male)이 암컷(female)보다 무겁고, 그 차이는 젠투펭귄에서 가장 크다

##상관관계 파악
지느러미 길이는 몸무게(body_mass_g)와 강한 양의 상관(0.87)이고
부리 깊이(bill_depth_mm)는 음의 상관(-0.47)을 보입니다.

    - 즉, 몸이 클수록 날개(지느러미)도 크며
    - 부리 깊이가 개체 크기가 아니라 종 특성을 반영하기 때문에 역방향 상관이 나타난 것으로 보인다.
'''



# 인사이트  <정밀ver.>
'''
##종별 몸무게 파악 (종(species) → 몸무게(body_mass_g))
젠투펭귄(Gentoo) 평균 5067.7g으로
아델리펭귄(Adelie, 3703.0g), 턱끈펭귄(Chinstrap, 3733.1g)보다 1300g 이상 무겁다.
아델리펭귄과 턱끈펭귄은 몸무게만으로는 거의 구분되지 않는다.

##섬별 몸무게 파악 (섬(island) → 몸무게(body_mass_g)) <교란변수 주의>
비스코섬(Biscoe) 평균 4712.1g으로 드림섬(Dream, 3712.9g), 토르거슨섬(Torgersen, 3713.0g)보다 훨씬 높다.
하지만 종(species) x 섬(island) 교차표를 보면 비스코섬=젠투펭귄 124마리 전부, 토르거슨섬=아델리펭귄 전부,
턱끈펭귄은 드림섬에만 존재해 섬과 종이 거의 1:1로 겹친다.
즉 "섬 효과"가 아니라 "그 섬에 어떤 종이 사는가"가 실질적 원인일 가능성이 크다.

##체급 구간에 따른 지느러미 길이 파악 (체급 구간(WeightBand) → 지느러미 길이(flipper_length_mm))
소형 188.4mm → 중소형 195.4mm → 중형 214.2mm → 대형 223.9mm로 단조 증가한다.
몸무게와 지느러미 길이가 함께 커지는 것은 상관관계 분석(0.87)과도 일치한다.

##성별 x 종별 몸무게 파악 (성별(sex) + 종(species) → 몸무게(body_mass_g))
세 종 모두 수컷(male)이 암컷(female)보다 무겁다.
성별 차이 폭: 젠투펭귄 729.0g > 아델리펭귄 642.9g > 턱끈펭귄 411.8g.
즉 성적 이형성(암수 체격차)이 젠투펭귄에서 가장 뚜렷하다.

##상관관계 파악 (부리 길이(bill_length_mm), 부리 깊이(bill_depth_mm), 지느러미 길이(flipper_length_mm), 몸무게(body_mass_g))
지느러미 길이는 몸무게와 강한 양의 상관(0.87) - 몸이 클수록 날개(지느러미)도 크다.
부리 깊이는 몸무게와 음의 상관(-0.47)인데, 이는 종 혼입 효과다.
젠투펭귄은 부리가 길면서도(47.5mm) 얕은(15.0mm) 반면 턱끈펭귄은 길고(48.8mm) 깊다(18.4mm) -
부리 깊이가 개체 크기가 아니라 종 특성을 반영하기 때문에 역방향 상관이 나타난 것으로 보인다.
'''
