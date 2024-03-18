import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 피처 엔지니어링
# 각 피처(변수,속성)을 하나씩 모델 학습에 맞도록 전처리 하고
# 모델 성능 개선할 수 있는 방법을 찾는 것을 말한다.

train = pd.read_csv("./Data/train.csv")
test = pd.read_csv("./Data/test.csv")
submission = pd.read_csv("./Data/submission.csv")

# 타겟변수의 분포확인
print(train['Survived'].value_counts(dropna=False))

# Survived
# 0    549
# 1    342

# 시본의 coutplot으로 생존자와 사망자의 분포를 시각화 해 봅니다.
# sns.countplot(x="Survived", data=train)
# plt.show()

print(train.columns)
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Pclass(객실등급)의 분포를 확인
# sns.countplot(x="Pclass", data=test)
# plt.show()

train['TrainSplit'] = "Train"
test['TrainSplit'] = "Test"
data = pd.concat([train, test], axis=0)

# sns.countplot(x="Pclass", data=data, hue="TrainSplit")
# plt.show()

# 객실 등급에 따른 생존자 비율을 확인
# sns.countplot(x="Pclass", data=train, hue="Survived")
# sns.countplot(x="Pclass", data=data[ data["TrainSplit"] =="Train" ], hue="Survived")
# plt.show()

# barplot함수를 이용하여 객실등급별 객실요금의 중간값(median)의 분포를 확인
# sns.barplot(data=train,x="Pclass", y="Fare")
# plt.show()

# barplot함수를 이용하여 객실등급별 객실요금의 중간값(median)의 분포를 확인 (생존여부 구분)
# sns.barplot(data=train,x="Pclass", y="Fare",hue="Survived")
# plt.show()

# sns.barplot(data=train,x="Pclass", y="Fare",hue="Survived", estimator=np.median)
# plt.show()

# 성별은 생존에 중요한 요인일까? 알아 봅시다.
# 성별별로 생존율 분포를 파악해 본다.
# sns.histplot(x="Sex", data=train, hue="Survived", multiple="dodge")
# plt.show()
# 그래프의 결과로써
# 남자는 사망자가 생존자의 수 보다 몇배이상 높게 나나타고
# 여자는 생존자가 사망자의 수 보다 두배이상 높게 나타나는 것으로 보아
# 성별은 생존여부에 중요한 요인인 것을 파악할 수 있어요.

# sns.histplot(x="Sex", data=train, hue="Survived", multiple="stack")
# plt.show()

# sns.histplot(x="Sex", data=train, hue="Survived", multiple="fill")
# plt.show()

# histplot에서
# hue 속성을 사용할 때에
# multiple 속성을  dodge라고 하면 각각을 분리하여 그려 해 주고
#                 stack으로 설정하면 누적하여 그려 해 주고
#                 fill로 설정하면 비율을 백분율로 그려 줍니다.


# 성별은 생존여부에 중요한 영향을 끼치는 속성으로 판별되었으니
# 학습에 참여시켜야 합니다.
# 그러기 위해서 현재 성별은 male, female값을 갖고 있는 문자열입니다.
# 이것을 숫자로 변경해 줘야 합니다.
# 문자인 피쳐를 숫자로 변경시키기 위한 방법은 다양할 수 있어요.
# 원-핫인코딩의 방법이 있을 수 도 있어요.
# 여기서는 직접 내손으로 female 0, male 1로 설정 해 봅시다.

data.loc[data['Sex'] == 'female', 'Sex'] = 0
data.loc[data['Sex'] == 'male', 'Sex'] = 1
data['Sex'] = data['Sex'].astype(int)

# 이름은 제각각 다를 텐데 이것을 포함시켜서 공부했을 때 점수가
# 이름을 제외시켰을때 점수보다 더 높아요.
# 이름의 값의 종류별 빈도를 출력
# print(data['Name'].value_counts())

# Connolly, Miss. Kate                                   2
# Kelly, Mr. James                                       2
# Braund, Mr. Owen Harris                                1
# Johnson, Master. Harold Theodor                        1
# Gustafsson, Mr. Alfred Ossian                          1

# 이름에는 신분을 나타내는
# Miss, Mr, Master 등이 있어요.
# 이것이 생존여부를 결정하는 중요한 요인이지 않을까 예상 해 볼 수 있어요.

# Connolly, Miss. Kate                                   2
# Kelly, Mr. James                                       2
# 위의 이름을 출력한 결과를 보면
# 쉼표 이후에 신분을 의미하는 글자가 나와요.
# 그래서 이름은 쉼표(,)로 분리하여 1번 인덱스를 갖고 오도록 합시다.

title_name = data["Name"].str.split(",",expand=True)[1]
# print(title_name)
# 0                                  Mr. Owen Harris
# 1       Mrs. John Bradley (Florence Briggs Thayer)
# 2                                      Miss. Laina
# 3               Mrs. Jacques Heath (Lily May Peel)
#

title = title_name.str.split(".",expand=True)[0]
# print(title)

# 0           Mr
# 1          Mrs
# 2         Miss
# 3          Mrs
# 4           Mr

# 각 신분별 데이터수를 확인하기
# print(title.value_counts(dropna=False))
# Mr              757
# Miss            260
# Mrs             197
# Master           61
# Rev               8
# Dr                8
# Col               4
# Mlle              2               'Mlle', 'the Countess', 'Lady', 'Don', 'Dona','Mme','Sir','Jonkheer'
# Major             2               'Col','Major','Capt'
# Ms                2               'Dr','Rev'
# Lady              1
# Sir               1
# Mme               1
# Don               1
# Capt              1
# the Countess      1
# Jonkheer          1
# Dona              1



# 비슷한 의미를 갖는 신분을 같은 그룹으로 묶어 줍니다.
# 문자열을 변경하는 replace함수를 이용합니다.

title = title.str.replace("Ms", "Miss")
# 유사한 의미를 갖는 Ms를 Miss로 변경하기

# title = title.str.replace(['Mlle', 'the Countess', 'Lady', 'Don', 'Dona','Mme','Sir','Jonkheer'], 'Noble')
# # 귀족을 칭하는 속성들을 모아서 모두  "Noble"로 묶어주기

for a in ['Mlle', 'the Countess', 'Lady', 'Don', 'Dona','Mme','Sir','Jonkheer', 'Noblea']:
    title = title.str.replace(a, 'Noble')

# title = title.str.replace(['Col','Major','Capt'], 'Officer')
# # 대령,소령,대위를 "장교"로 묶어주기
for a in ['Col','Major','Capt']:
    title = title.str.replace(a, 'Officer')

# title = title.str.replace(['Dr','Rev'], 'Priest')
# 의사,목사를 중요한 인물로 묶어주기
for a in ['Dr','Rev']:
    title = title.str.replace(a, 'Priest')

print(title.value_counts())


# 만들어진 신분을 데이터프레임에 Title속성으로 추가 합니다.
data['Title'] = np.array(title)
print(data['Title'].value_counts(dropna=False))
print(data.head())

# Title의 종류별 빈도수를 확인
print(data["Title"].value_counts(dropna=False))
# Mr         757
# Miss       262
# Mrs        197
# Master      61
# Priest      16
# Noble        9
# Officer      7

# 신분(Title)별로 승객나이 별로 생존여부의 분포를 바이올린플랏으로 확인
# sns.violinplot(data=data, x="Title", y="Age", hue="Survived",split=True)
# plt.show()

# sns.violinplot(data=data, x="Title", y="Age", hue="Survived", split=True)
# plt.show()

print(data.columns)

# 중복된 열
print(data.columns.duplicated())

print("중복된 행 ..............................................")
# print(data.index.duplicated())

idx = np.arange(len(data))
print(idx)
data.index = idx

# sns.violinplot(data=data, x="Title", y="Age", hue="Survived", split=True)
# plt.show()

# Name 변수(속성, 열)을 삭제하기
data.drop("Name", axis=1, inplace=True)

print(data.columns)

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'TrainSplit', 'Title'],
#       dtype='object')

# Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
#        'Ticket', 'Fare', 'Cabin', 'Embarked', 'TrainSplit', 'Title'],
#       dtype='object')


# 변수 나이의 결측치의 수를 확인
print(data["Age"].isna().sum())
# 263

# 나이에 대한 결측치를 각 신분(Title)별로 중앙값을
# 구하여 설정하도록 합시다.
print(data["Title"].unique())
# [' Mr' ' Mrs' ' Miss' ' Master' ' Noble' ' Priest' ' Officer']

# for title in data["Title"].unique():
#     age_median = data.loc[data["Title"] == title, "Age"].median()
#     print(title, age_median)
#
#  Mr 29.0
#  Mrs 35.5
#  Miss 22.0
#  Master 4.0
#  Noble 38.0
#  Priest 44.0
#  Officer 53.0


# for title in data["Title"].unique():
#     age_median = data.loc[data["Title"] == title, "Age"].median()
#     print( title,   data.loc[data["Title"] == title, "Age"].isna().sum())

 # Mr 176
 # Mrs 27
 # Miss 51
 # Master 8
 # Noble 0
 # Priest 1
 # Officer 0

for title in data["Title"].unique():
    age_median = data.loc[data["Title"] == title, "Age"].median()
    data.loc[data["Title"] == title, "Age"] = data.loc[data["Title"] == title, "Age"].fillna(age_median)

print(data['Age'].isna().sum())
# 0

# 시본의 displot함수를 이용하여 나이별로 생존여부의 분포를 확인해 봅시다.
# sns.displot(data=data[data["TrainSplit"] == "Train"] ,
#             x="Age",
#             kind="hist",
#             hue="Survived")
# plt.show()

# 나이에 따라 생존율에 차이가 크기 때문에 이것을 학습데이터로
# 적용시키기 위하여 나이를 그대로 학습시키기 보다는 나이을 몇개의 구간으로 나누어 봅시다.
# 데이터를 구간으로 나누는 기법
# ===> 비닝(Binning) 기법
# pandas cut함수를 이용하고
#       bins        구간경계값 리스트       n +1        [0,50,100]          <--- 두개의 구간
#       labels      구간이름의 리스트       n           [구간1, 구간2]

# 나이를 10개의 구간으로 나누르면
# 구간경계값의 배열     ==>  11개
bins = [0,4,8,12,16,32,36,48,56,64,100]

# 구간이름의 배열      ==> 10개
labels = ["Infant", 'Child1','Child2','Youth1','Youth2','Adult1','Adult2','Middle Aged','Senior','Elerly']

# 위에서 작성한 구간경계값과 구간이름의 배열을 가지고 데이터프레임에 나이를 구간으로 나눈 파생변수를 추가 합니다.
data["AgeBin"] = pd.cut(data["Age"], bins=bins, labels=labels)

print(data.loc[:,["Age","AgeBin"]].head())
#     Age  AgeBin
# 0  22.0  Youth2
# 1  38.0  Adult2
# 2  26.0  Youth2
# 3  35.0  Adult1
# 4  35.0  Adult1

# 나이구간(AgeBin)에 따른 생존율을 비교
# sns.countplot(x="AgeBin",
#               hue="Survived",
#               data=data[data["TrainSplit"] == "Train"])
# plt.xticks(rotation=90)
# plt.show()

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'TrainSplit', 'Title'],
#       dtype='object')

# SibSp 동승한 형제,자매,배우자의 수
print(data.loc[:,['PassengerId','SibSp']])
#       PassengerId  SibSp
# 0               1      1
# 1               2      1
# 2               3      0
# 3               4      1
# 4               5      0

# 동승한 형제자매,배우자의 수 별로 나이별로 생존율을 파악
# sns.boxplot(x="SibSp", y="Age", hue="Survived",
#             data=data[data["TrainSplit"]=="Train"])
# plt.show()


# Parch     함께 탑승한 부모/자식의 수
# 함께 탑승한 부모자식의 수 별로 나이별로 생존율의 관계
# sns.boxplot(x="Parch", y="Age", hue="Survived",
#             data=data[data["TrainSplit"]=="Train"])
# plt.show()

# 함께탑승한 형제자매의수+함께탑승한 부모자녀의 수 + 1을 하여 전체 가족구성원 수를 의미하는 파생변수를 만들어요.
data["FamilySize"] = data["SibSp"] +  data["Parch"] + 1
print(data.head())
print(data.loc[:,['PassengerId','SibSp','Parch','FamilySize']].head())
print(data.loc[:,['PassengerId','SibSp','Parch','FamilySize']].tail())

#    PassengerId  SibSp  Parch  FamilySize
# 0            1      1      0           2
# 1            2      1      0           2
# 2            3      0      0           1
# 3            4      1      0           2
# 4            5      0      0           1

# 가족구성원의 수와 생존율의 관계
# sns.barplot(
#     x="FamilySize",
#     y="Survived",
#     hue="Pclass",
#     data=data[data["TrainSplit"] == "Train"]
# )
# plt.show()

# 요금에 결측치 확인하기
print(data["Fare"].isna().sum())
# 1

# 요금에 결측치가 1개 있어요!
print(data.loc[ data["Fare"].isnull() ,['PassengerId','Pclass','Fare']])

#       PassengerId  Pclass  Fare
# 1043         1044       3   NaN

# 이사람의 객실등급이 3등급 이니깐
# 3등급의 평균요금을 구해서 그 값으로 설정하자.

p3_fare_mean = data.loc[data["Pclass"] == 3, "Fare"].mean()
print(p3_fare_mean)
# 13.302888700564973

data['Fare'] = data['Fare'].fillna(p3_fare_mean)
print(data.loc[ data["PassengerId"] == 1044 ,['PassengerId','Pclass','Fare']])
#       PassengerId  Pclass       Fare
# 1043         1044       3  13.302889


# 요금(Fare)의 분포를 그래프로 나타내 봅니다.
# sns.displot(x="Fare", data=data[data["TrainSplit"]=="Train"], hue="Survived", kind="kde")
# plt.show()

# 결과를 확인해보면
# 요금은 왼쪽으로 치우친 비대칭 분포를 보입니다.
# 비대칭 분포를 가진 데이터를 학습효율이 떨어집니다.
# 이것을 대칭 분포를 가지도록 변환(정규화) --> 로그변환(np.log1p   <---- 숫자1 )

data['FareLog'] = np.log1p(data['Fare'])
# sns.displot(
#     x = "FareLog",
#     data=data[data["TrainSplit"]=="Train"],
#     hue="Survived",
#     kind="hist"
# )
# plt.show()
# 결과를 확인해 보면
# 가장 빈도가 많은 데이터가 원래는 왼쪽으로 치우친 비대칭이었는데
# 가장 빈도가 많은 데이터를 중앙에 오도록 대칭형태로 만들어진것을 확인할 수 있어요.
# 이래야 머신러닝이 학습을 잘 해요.

# 시본 stripplot
# 각 카테고리별 데이터를 수평, 혹은 수직으로 점을찍어 표현하여
# 데이터의 분포를 쉽게 파악할 수 있어요.

# 객실등급과 요금간의 관계를 시각화 해 봅시다.
# sns.stripplot(
#     x="Pclass",
#     y="FareLog",
#     hue="Survived",
#     data=data[data["TrainSplit"] == "Train"]
# )
# plt.show()


# Embarked(탑승항구) 결측치 확인
print(data["Embarked"].isnull().sum())
# 2

print(data.loc[ data["Embarked"].isnull(), ['PassengerId','Pclass','Fare', 'Embarked']])
#      PassengerId  Pclass  Fare Embarked
# 61            62       1  80.0      NaN
# 829          830       1  80.0      NaN

# 최빈값으로 변환하기
print("탑승항구의 최빈값:",data['Embarked'].mode())
# 탑승항구의 최빈값: 0    S

# 값(value)만 뽑아오기 위해서 인덱싱을 해 줍니다.
print("탑승항구의 최빈값:",data['Embarked'].mode()[0])
# 탑승항구의 최빈값: S

data["Embarked"] = data['Embarked'].fillna( data["Embarked"].mode()[0])

# 탑승항구별로 빈도수를 출력
print(data["Embarked"].value_counts(dropna=False))
# S    916
# C    270
# Q    123

# 시본의 catplot
# 범주형 데이터를 시각화 하는데 용이한 함수 입니다.
# 범주형 데이터( 혈액형 - A,B,AB,O,
#                탑승항구 - S, C, Q)

# sns.catplot(
#     data=data[data["TrainSplit"] == "Train"],
#     x="Embarked",
#     y="Survived",
#     kind="point"
# )
# plt.show()
# kind="point", "bar","violin"
print("OK")


# Cabin : 객실 번호
print(data["Cabin"].isna().sum())
# 1014
print(data['Cabin'].unique())
# [nan 'C85' 'C123' 'E46' 'G6' 'C103' 'D56' 'A6' 'C23 C25 C27' 'B78' 'D33'
#  'B30' 'C52' 'B28' 'C83' 'F33' 'F G73' 'E31' 'A5' 'D10 D12' 'D26' 'C110'

# 객실번호의 1글자를 잘라와서 객실구역으로 활용해 봅시다.
print(data['Cabin'].str.slice(0,1).value_counts(dropna=False))

# NaN    1014
# C        94
# B        65
# D        46
# E        41
# A        22
# F        21
# G         5
# T         1

# 객실변호를 객실번호의 첫글자를 찾아서 등급으로 바꿉니다.
# 결측치는  U로 설정합니다.
data["Cabin"] = data["Cabin"].str.slice(0,1)
data["Cabin"] = data["Cabin"].fillna("U")

print(data["Cabin"].value_counts())
# U    1014
# C      94
# B      65
# D      46
# E      41
# A      22
# F      21
# G       5
# T       1

# catplot함수를 이용하여 객실구역(Cabin) 별로 생존율 비교
# sns.catplot(
#     x="Cabin",
#     y="Survived",
#     kind="bar",
#     data=data[data["TrainSplit"] == "Train"]
# )
# plt.show()


# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'TrainSplit', 'Title'],
#       dtype='object')

print(data["Ticket"].head(20))
print(data["Ticket"].value_counts(dropna=False))

# 0            A/5 21171
# 1             PC 17599
# 2     STON/O2. 3101282
# 3               113803
# CA. 2343        11
# CA 2144          8
# 1601             8
# PC 17608         7
# S.O.C. 14879     7
#                 ..
# 113792           1
# 36209            1
# 323592           1
# 315089           1
# 359309           1

# 탑승권의 글자중에 쩜(.)과 슬러시(/)를 지웁시다.
data["Ticket"] = data["Ticket"].str.replace(".","").str.replace("/","")

print(data["Ticket"].head(20))
print(data["Ticket"].value_counts(dropna=False))
# 0           A5 21171
# 1           PC 17599
# 2     STONO2 3101282
# 3             113803
# Ticket
# CA 2343      11
# CA 2144       8
# 1601          8
# PC 17608      7
# SOC 14879     7

# 티켓의 글자를 공백(" ")로 분리하여 0번재 인덱스를 갖고 오고
data["Ticket"] = data["Ticket"].str.strip().str.split(" ").str[0]
" ABC 123 "
# strip() ==> 문자열의 맨 앞뒤의 공백을 제거
# print(data["Ticket"].head(20))
# print(data["Ticket"].value_counts(dropna=False))

# 숫자로만 구성이 된것은 "NUM"로 설정합니다.
data.loc[data["Ticket"].str.isdigit(), "Ticket"] = "NUM"
#
# print(data["Ticket"].head(20))
# print(data["Ticket"].value_counts(dropna=False))

# 0         A5
# 1         PC
# 2     STONO2
# 3        NUM
# 4        NUM
# 5        NUM

# Ticket
# NUM        957
# PC          92
# CA          68
# A5          28
# SOTONOQ     24
# WC          15

# sns.catplot(x="Ticket",y="Survived",kind="bar",data=data[data["TrainSplit"]=="Train"])
# plt.xticks(rotation=90)
# plt.show()


#label인코딩 : 문자를 숫자로 변경하는것을 말하며 sklearn.preprocessing.LabelEncoder를 이용합니다.
#앞에서 우리는 성별에 대하여 직접 Label인코딩을 해봤습니다. (Female = 0 , male = 1)을 갖도록 설정하였습니다.
#그런데 신분을 의미하는 Title은 종류가 많아요. 일일이 하나하나 00,1,2,..로 설정하는 것은 번거로워요
#알아서 설정 해주는 패키지 함수 LabeleEncoder를 이용해봅시다.

print(data.loc[:,["Title","AgeBin"]].head())

# Title  AgeBin
# 0     Mr  Youth2
# 1    Mrs  Adult2
# 2   Miss  Youth2
# 3    Mrs  Adult1
# 4     Mr  Adult1

from sklearn.preprocessing import LabelEncoder
for col in ['Title','AgeBin']:
    encoder = LabelEncoder()
    data[col]=encoder.fit_transform(data[col])

# Title  AgeBin
# 0      2       9
# 1      3       1
# 2      1       9
# 3      3       0
# 4      2       0
# print(data.loc[:,["Title","AgeBin"]].head())

# 원핫인코딩
# 문자열데이터를 숫자로 변환하는 방법 중의 하나이며
# 값의 종류 수 만큼 칼럼을 추가해서
# 해당하는 곳에만 불을 키고(1을설정) 나머지 칼럼에는 모두 불을 끄는(0으로설정) 방식입니다
# pandas.get_dummies함수를 이용하며 drop_first=True 옵션을 주면 첫번째 열을 삭제합니다.
# 하나의 열이 없어도 범주는 구분하는데 충분합니다.
# 아래의 탑승항구의 종류가 C,S,Q일때
# C컬럼이 없어도 S,Q만으로도 충분히 표현할 수 있어요
# S에도 불이꺼지고 Q에도 불이 꺼지면 C로 볼 수 있어요.

#범주형 데이터인 탑승항구,Cabin,Ticket에 대하여 원핫 인코딩으로 변환 해 봅시다.
for col in ["Embarked","Cabin","Ticket"]:
    data[col]= data[col].astype("category")
    data =pandas.get_dummies(data,columns=[col],prefix=col[:3],drop_first=True)


#   Emb_Q  Emb_S
# 0  False   True
# 1  False  False
# 2  False   True
# 3  False   True
# 4  False   True
# print(data.loc[:,["Emb_Q","Emb_S"]].head())

#피쳐스케일링
#학습을 시킬 피쳐들의 범위가 차이가 나면 학습 성능이 좋지 않을 수 있어요.
#각 피쳐들의 값의 범위를 동일한 사이즈로 맞출 필요가 있어요.
#학습에 필요한 피쳐들의 범위를 0~1 사이에 값으로 맞추어요 ==> 정규화
#sklearn.preprocessing.MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()

# 전체데이터로부터 스케일링 할 피쳐들만 추출합니다.
# 전체데이터에는 학습용 데이터인지 테스트 데이터인지 구별할 TrainSplit 피쳐는 제외시킵니다.
scaled_cols =  [col for col in data.loc[:,"Pclass":].columns if col!="TrainSplit"]

# print(scaled_cals)

#원본으로부터 스케일링 할 대상이 되는 데이터만 일단 갖고와요.
data_scaled = data.loc[:,scaled_cols] #아직 스케일 안된상태.

#스케일링을 합니다
data_scaled=scaler.fit_transform(data_scaled)


#스케일링된 데이터를 원본에 설정합니다.
# data.loc[:,scaled_cols] = data_scaled[:,:] //오류가 나서 새로운 데이터프레임을 만들어 스케일된 데이터를 넣음
df = pd.DataFrame(data_scaled,columns=scaled_cols)
# print(df.head())

#여기에 원본으로 부터 Survived, TrainSplit을 추가하도록 합시다.
df["Survived"] = data.loc[:,"Survived"]
df["TrainSplit"] = data.loc[:,"TrainSplit"]


selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'AgeBin',
       'FamilySize', 'FareLog', 'Emb_Q', 'Emb_S', 'Cab_B', 'Cab_C', 'Cab_D',
       'Cab_E', 'Cab_F', 'Cab_G', 'Cab_T', 'Cab_U', 'Tic_A4', 'Tic_A5',
       'Tic_AQ3', 'Tic_AQ4', 'Tic_AS', 'Tic_C', 'Tic_CA', 'Tic_CASOTON',
       'Tic_FC', 'Tic_FCC', 'Tic_Fa', 'Tic_LINE', 'Tic_LP', 'Tic_NUM',
       'Tic_PC', 'Tic_PP', 'Tic_PPP', 'Tic_SC', 'Tic_SCA3', 'Tic_SCA4',
       'Tic_SCAH', 'Tic_SCOW', 'Tic_SCPARIS', 'Tic_SCParis', 'Tic_SOC',
       'Tic_SOP', 'Tic_SOPP', 'Tic_SOTONO2', 'Tic_SOTONOQ', 'Tic_SP',
       'Tic_STONO', 'Tic_STONO2', 'Tic_STONOQ', 'Tic_SWPP', 'Tic_WC',
       'Tic_WEP'] #56개

#전체 데이터로 부터 훈련데이터의 답안을 뽑아 옵니다.
y_train = df.loc[data["TrainSplit"]=="Train","Survived"]
#전체 데이터로 부터 훈련데이터의 문제를 뽑아 옵니다.
x_train = df.loc[data["TrainSplit"]=="Train",selected_features]
#전체 데이터로 부터 테스트데이터의 문제를 뽑아 옵니다.
x_test = df.loc[data["TrainSplit"]=="Test",selected_features]

# (891, 56) (891,)
# (418, 56)
# print(x_train.shape,y_train.shape)
# print(x_test.shape)

# train데이터로 부터 훈련데이터와 검증데이터로 분리합니다.
from sklearn.model_selection import train_test_split

X_tr, X_val,y_tr, y_val = train_test_split(x_train,y_train,test_size=0.2 ,shuffle=True, random_state=1234)

# 훈련데이터셋 크기 :   (712, 56) (712,)
# 검증데이터셋 크기 (179, 56) (179,)
# print("훈련데이터셋 크기 :  ",X_tr.shape, y_tr.shape)
# print("검증데이터셋 크기" , X_val.shape, y_val.shape)

#로지스틱 회귀 분류 모델을 이용하여 학습시키고 검증 해 봅시다.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_tr,y_tr)

#공부한 문제를 예측시키기
y_tr_pred=model.predict(X_tr)
#공부안한 문제를 예측시키기
y_val_pred=model.predict(X_val)

from sklearn.metrics import accuracy_score

# 공부한 문제의 정확도 :  0.8202247191011236
# 공부안한 문제의 정확도 :  0.8324022346368715
# print("공부한 문제의 정확도 : ",accuracy_score(y_tr,y_tr_pred))
# print("공부안한 문제의 정확도 : ",accuracy_score(y_val,y_val_pred))

# 성능지표(정확도,정밀도,재현율, f1-score)
# ROC-AUC score (Receiver Operaing Characteristic-Area Under the Curve)
# ==> 이진분류(yes or no ) 모델의 성능평가의 지표중의 하나입니다.
# 데이콘은 ROC-AUC score를 성능지표로 사용한다.


from sklearn.metrics import roc_auc_score
# 공부한 문제의 roc_score :  0.8026069518716579
# 공부안한 문제의 roc_score :  0.8112712975098297
# print("공부한 문제의 roc_score : ",roc_auc_score(y_tr,y_tr_pred))
# print("공부안한 문제의 roc_score : ",roc_auc_score(y_val,y_val_pred))


#공부한 모델을 갖고 시험문제 풀이를 시킵니다.
y_test_pred = model.predict(x_test)

#예측결과를 답안파일을 읽어들인 데이터프레임(submission)에 담습니다
submission['Survived'] = y_test_pred.astype(int)

#예측한결과(시험문제 풀이 결과)를 답안 파일로 저장합니다.
# submission.to_csv("./Data/titanic_0318_1.csv",index=False)
# print("저장이 완료되었습니다")








