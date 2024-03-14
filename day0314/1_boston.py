import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(type(data))
print(type(target))
# <class 'numpy.ndarray'>
# <class 'numpy.ndarray'>

print(data.shape)
print(target.shape)
# (506, 13)
# (506,)

# data : 문제
# taget : 답

# data를 갖고 데이터프레임을 만들어요
df = pd.DataFrame(data)
print(df.head())

# 데이터프레임에 칼럼이름 정하기
# ['CRIM',      범죄율
# 'ZN',         토지비율
# 'INDUS',      상업지역비율
# 'CHAS',       찰스강 인접여부 1 0
# 'NOX',        일산화질소 농도
# 'RM',         방의 수
# 'AGE',        자가 주택 비율
# 'DIS',        고용지역 거리
# 'RAD',        고속도로 접근성
# 'TAX',        재산세
# 'PTRATIO',    교사학생비율
# 'B',          흑인비율
# 'LSTAT']      저소득층비율

df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
print(df.head())

# 데이터프레임에 target을 설정합니다.
df_target = pd.DataFrame(target)
df_target.columns = ["Target"]
print(df_target.head())

df = pd.concat([df, df_target],axis=1)
print(df.head())

#      CRIM    ZN  INDUS  CHAS    NOX  ...    TAX  PTRATIO       B  LSTAT  Target
# 0  0.00632  18.0   2.31   0.0  0.538  ...  296.0     15.3  396.90   4.98    24.0
# 1  0.02731   0.0   7.07   0.0  0.469  ...  242.0     17.8  396.90   9.14    21.6
# 2  0.02729   0.0   7.07   0.0  0.469  ...  242.0     17.8  392.83   4.03    34.7
# 3  0.03237   0.0   2.18   0.0  0.458  ...  222.0     18.7  394.63   2.94    33.4
# 4  0.06905   0.0   2.18   0.0  0.458  ...  222.0     18.7  396.90   5.33    36.2

# 데이터 탐색
print(df.info())

# RangeIndex: 506 entries, 0 to 505
# Data columns (total 14 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   CRIM     506 non-null    float64
#  1   ZN       506 non-null    float64
#  2   INDUS    506 non-null    float64
#  3   CHAS     506 non-null    float64
#  4   NOX      506 non-null    float64
#  5   RM       506 non-null    float64
#  6   AGE      506 non-null    float64
#  7   DIS      506 non-null    float64
#  8   RAD      506 non-null    float64
#  9   TAX      506 non-null    float64
#  10  PTRATIO  506 non-null    float64
#  11  B        506 non-null    float64
#  12  LSTAT    506 non-null    float64
#  13  Target   506 non-null    float64
# dtypes: float64(14)
# memory usage: 55.5 KB
# None

# info함수의 결과로써
# 모든속성에 대하여 결측치가 하나도 없고
# 모든속성 숫자 자료형이므로 원핫 인코딩이 필요없어요.


#결측치를 확인
print(df.isna().sum())
print(df.isnull().sum())
# CRIM       0
# ZN         0
# INDUS      0
# CHAS       0
# NOX        0
# RM         0
# AGE        0
# DIS        0
# RAD        0
# TAX        0
# PTRATIO    0
# B          0
# LSTAT      0
# Target     0

print("-"*50)
#     각 변수끼리 상관행렬
df_corr= df.corr()
print(df_corr)
'''
             CRIM        ZN     INDUS  ...         B     LSTAT    Target
CRIM     1.000000 -0.200469  0.406583  ... -0.385064  0.455621 -0.388305
ZN      -0.200469  1.000000 -0.533828  ...  0.175520 -0.412995  0.360445
INDUS    0.406583 -0.533828  1.000000  ... -0.356977  0.603800 -0.483725
CHAS    -0.055892 -0.042697  0.062938  ...  0.048788 -0.053929  0.175260
NOX      0.420972 -0.516604  0.763651  ... -0.380051  0.590879 -0.427321
RM      -0.219247  0.311991 -0.391676  ...  0.128069 -0.613808  0.695360
AGE      0.352734 -0.569537  0.644779  ... -0.273534  0.602339 -0.376955
DIS     -0.379670  0.664408 -0.708027  ...  0.291512 -0.496996  0.249929
RAD      0.625505 -0.311948  0.595129  ... -0.444413  0.488676 -0.381626
TAX      0.582764 -0.314563  0.720760  ... -0.441808  0.543993 -0.468536
PTRATIO  0.289946 -0.391679  0.383248  ... -0.177383  0.374044 -0.507787
B       -0.385064  0.175520 -0.356977  ...  1.000000 -0.366087  0.333461
LSTAT    0.455621 -0.412995  0.603800  ... -0.366087  1.000000 -0.737663
Target  -0.388305  0.360445 -0.483725  ...  0.333461 -0.737663  1.000000
'''

#  상관행렬을 히트맵으로 그려보기
import seaborn as sns
# plt.figure(figsize=(10,10))
# sns.set(font_scale=0.8)
# sns.heatmap(df_corr, annot=True, cbar=False)
# plt.show()

#  집값에 가장 영향을 많이 끼치는 변수를 확인
#  목표변수인 Target과 상관계수가 높은순서대로 출력
# corr_order = df.corr().loc[:'LSTAT', 'Target']
# print(corr_order)
# [14 rows x 14 columns]
# CRIM      -0.388305
# ZN         0.360445
# INDUS     -0.483725
# CHAS       0.175260
# NOX       -0.427321
# RM         0.695360
# AGE       -0.376955
# DIS        0.249929
# RAD       -0.381626
# TAX       -0.468536
# PTRATIO   -0.507787
# B          0.333461
# LSTAT     -0.737663

corr_order = df.corr().loc[:'LSTAT', 'Target'].abs().sort_values(ascending=False)
print(corr_order)
# LSTAT      0.737663       저소득층비율
# RM         0.695360       방의 갯수
# PTRATIO    0.507787       교사학생비율
# INDUS      0.483725       상업지역비율
# TAX        0.468536
# NOX        0.427321
# CRIM       0.388305
# RAD        0.381626
# AGE        0.376955
# ZN         0.360445
# B          0.333461
# DIS        0.249929
# CHAS       0.175260


# 타겟(집값)에 가장 영향을 많이 끼치는 4개의 피쳐(속성)만 뽑아오도록 합시다.
plot_cols = ['Target','LSTAT','RM','PTRATIO','INDUS']
plot_df = df.loc[:, plot_cols]
print(plot_df.shape)
# (506, 4)

print(plot_df.head())
#    LSTAT     RM  PTRATIO  INDUS
# 0   4.98  6.575     15.3   2.31
# 1   9.14  6.421     17.8   7.07
# 2   4.03  7.185     17.8   7.07
# 3   2.94  6.998     18.7   2.18
# 4   5.33  7.147     18.7   2.18

print(plot_df['RM'].mean())
# 6.284634387351779

# 집값을 결정하는 중요한 변수 4개에 대하여
# 각각 산점도를 그려보도록 합니다.
# 산점도를 그리고 그리고 그 데이터를 잘 설명할 수 있는 선도 같이 그려봅니다.
# sns.regplot

# for col in plot_cols[1:]:
#     print(col)

# 루프를 돌면서 나는 인덱스로 필요해
# enumerate

# for i, col in enumerate(plot_cols[1:]):
#     print(i, col)
# 0 LSTAT
# 1 RM
# 2 PTRATIO
# 3 INDUS

# plt.figure(figsize=(10,10))               #도화지 크기 설정
# for i, col in enumerate(plot_cols[1:]):   #컬럼이름을 담고 있는 리스트의 1번째 요소부터 반복, 인덱스
#     ax1 = plt.subplot(2,2,i+1)            #화면을 2행 2열로 분할하여 i+1번째 화면을 선택
#     sns.regplot(x=col, y=plot_cols[0],data=plot_df, ax=ax1 )
# plt.show()

# 반복문을 이용하지 않고 만들어 보기
# plt.figure(figsize=(10,10))
# f1 = plt.subplot(2,2,1)
# f2 = plt.subplot(2,2,2)
# f3 = plt.subplot(2,2,3)
# f4 = plt.subplot(2,2,4)
# sns.regplot(x='LSTAT', y='Target', data=plot_df, ax=f1)
# sns.regplot(x='RM', y='Target', data=plot_df, ax=f2)
# sns.regplot(x='PTRATIO', y='Target', data=plot_df, ax=f3)
# sns.regplot(x='INDUS', y='Target', data=plot_df, ax=f4)
# plt.show()

# 저소득층이 높을수록 집값은 떨어지고
# 방의 수가 많을수록 집값이 높아지는 것을 확인할 수 있어요.

# 집값에 대한 데이터 분포도를 그려 봅니다.
# sns.displot(x='Target', kind='hist',data=df)
# sns.displot(x='Target', kind='kde',data=df)
# plt.show()

print(df.head())
print(df.columns)

print(plot_df.describe())
#
#            Target       LSTAT          RM     PTRATIO       INDUS
# count  506.000000  506.000000  506.000000  506.000000  506.000000
# mean    22.532806   12.653063    6.284634   18.455534   11.136779
# std      9.197104    7.141062    0.702617    2.164946    6.860353
# min      5.000000    1.730000    3.561000   12.600000    0.460000
# 25%     17.025000    6.950000    5.885500   17.400000    5.190000
# 50%     21.200000   11.360000    6.208500   19.050000    9.690000
# 75%     25.000000   16.955000    6.623500   20.200000   18.100000
# max     50.000000   37.970000    8.780000   22.000000   27.740000

print(df['B'].describe())

# 피쳐 스케일링
# 집값의 범위는 5~50, 저소득층 비율의 값의 범위 1~37, 방의 개수는 3.56~8.78~~,
# 흑인 비율 0~396 으로 확인됩니다.
# 지금의 경우는 값의 단위가 그다지 크다고 볼 수는 없지만
# 실제로 현장에서는
# 어떤 속성의 값의 범위가 0~1 사이이고
# 어떤 속성의 값의 범위는 10000~50000 일 수도 있어요.
# 이상태로 학습을 시키면 학습효율이 떨어지기 때문에
# 각 변수로 값의 범위(scale)을 맞출 필요가 있어요.
# 보통는 모든 속성을 0~1으로 변환시켜서 학습시킵니다.
# 이러한 작업을 "정규화" 혹은 "피쳐 스케일링"이라고 합니다.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 타겟변수(집값)을 제외한 속성들을 뽑아 옵니다.
df_scaler = df.iloc[:,:-1]
print(df_scaler.head())

# 스케일할 준비를 시켜요.
scaler.fit(df_scaler)

# 준비한 대로 스케일링을 적용시켜
df_scaler = scaler.transform(df_scaler)

print(type(df_scaler))
# <class 'numpy.ndarray'>
print(df_scaler[:,:])
df.iloc[:,:-1] = df_scaler[:,:]
print(df.head())

#      CRIM    ZN  INDUS  CHAS    NOX  ...  RAD    TAX  PTRATIO       B  LSTAT
# 0  0.00632  18.0   2.31   0.0  0.538  ...  1.0  296.0     15.3  396.90   4.98
# 1  0.02731   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  396.90   9.14
# 2  0.02729   0.0   7.07   0.0  0.469  ...  2.0  242.0     17.8  392.83   4.03
# 3  0.03237   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  394.63   2.94
# 4  0.06905   0.0   2.18   0.0  0.458  ...  3.0  222.0     18.7  396.90   5.33

#        CRIM    ZN     INDUS  CHAS  ...   PTRATIO         B     LSTAT  Target
# 0  0.000000  0.18  0.067815   0.0  ...  0.287234  1.000000  0.089680    24.0
# 1  0.000236  0.00  0.242302   0.0  ...  0.553191  1.000000  0.204470    21.6
# 2  0.000236  0.00  0.242302   0.0  ...  0.553191  0.989737  0.063466    34.7
# 3  0.000293  0.00  0.063050   0.0  ...  0.648936  0.994276  0.033389    33.4
# 4  0.000705  0.00  0.063050   0.0  ...  0.648936  1.000000  0.099338    36.2

# 연습) 저소득비율과 방의수에 따른 집값을 예측하는 모델을 만들어 보세요.
#       선형회귀 모델을 이용합니다.
#       모델을 생성하고 학습을 시키고 정확도를 판별 해 봅니다.

# 전체데이터로 부터 학습데이터와 테스트데이를 분리시키기 위한
# train_test_split함수를 사용하기 위한 패키지를 포함시킨다.
from sklearn.model_selection import train_test_split

# 전체데이터로 부터 학습시킬 문제에 해당하는 속성만 추출한다.
X_data = df.loc[:,["LSTAT","RM"]]

# 전체데이터 부터 학습시킬 답에 해당하는 속성을 추출한다.
y_data = df.loc[:,"Target"]

print(type(X_data))
print(type(y_data))
# <class 'pandas.core.frame.DataFrame'>
# <class 'pandas.core.series.Series'>

# 학습문제, 테스트문제, 학습답, 테스트답으로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(
                                            X_data, y_data,
                                            test_size=0.2, random_state=1234, shuffle=True)
# 학습할 문제로 사용되는
# 두개의 변수(저소득층 비율과 방의 수)는 타켓(집값)과의
# 산점도, 회귀선을 그려보았을때 강한 상관관계에 있으므로 (두 데이터 사이의 관계를 직선으로 나타낼 수 있음)
# 학습시킬 모델로는 "선형 회귀 모델"이 적합하겠습니다.

# 선형회귀모델을 사용하기 위하여 포함시킵니다.
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# y = a1*x1 +  a2*x2 + b
# 공부한 결과 위와 같은 회귀식을 잘 나타내기 위한
# a1, a2, b를 구하기 될 것입니다.
# x1는 저소득비율인 LSTAT,  x2는 방의 수 인 RM이 됩니다.
# 이것이 학습의 목적이 a1, a2, b를 찾아내는 것입니다.
# a1, a2    ===> 회귀계수(기울기)          ==> coef_
# b         ===> 상수항(절편)             ==> intercept_

print("회귀계수(기울기) :", model.coef_)
print("상수항(절편) :", model.intercept_)

# 회귀계수(기울기) : [-23.92919139  22.48267361]
# 상수항(절편) : 17.99369369045411
# y = -23.92919139*LSTAT +  22.48267361*RM + 17.99369369045411
# 학습한결과 로써 위의 함수를 만들어 내고
# 새로운데이터 저소득비율과 방의 수 가 주어지면
# 위의 수식을 거쳐 집값을 예측할 수 있다.

# 예측시키기
y_pred = model.predict(X_test)

print(type(y_pred))  #예측한 답의 자료형
print(type(y_test))  #진짜 답의 자료형
# <class 'numpy.ndarray'>
# <class 'pandas.core.series.Series'>

print(y_pred[:5])
print(y_test.iloc[:5])
# [29.08332733 26.5546596  11.91068356 24.00095608 20.42480112]
# 64     33.0
# 100    27.5
# 400     5.6
# 485    21.2
# 454    14.9

# 위와 같은 데이터는
# 오차의 범위를 가장 작게하는 모델을 찾는 것이 중요합니다.


print("-"*50)
# 공부한 데이터 예측시키기
y_train_pred = model.predict(X_train)
print(y_train_pred[:5])
print(y_train.iloc[:5])


# X_data = df.loc[:,["LSTAT","RM"]]
# 실제값과 예측값을 산점도를 그려서 확인 해 봅시다.
# plt.figure(figsize=(10,5))
# plt.scatter(X_test['LSTAT'], y_test, label='y_test')
# plt.scatter(X_test['LSTAT'], y_pred, label='y_pred', c='r')
# plt.legend(loc="best")
# plt.show()

# test = X_test.iloc[:2,:]
# plt.figure(figsize=(10,5))
# plt.scatter(test['LSTAT'], y_test[:2], label='y_test')
# plt.scatter(test['LSTAT'], y_pred[:2], label='y_pred', c='r')
# plt.legend(loc="best")
# plt.show()

# 선형회귀의 공부를 잘 했는지 성능을 파악하기 위해서는
# 실제값과 예측값의 차이의 제곱의 합의 평균인 MSE를 이용합니다.
# 훈련데이터의 MSE와 테스트데이터의 MSE의 차이가 없다면 학습을 잘 했다고 볼 수 있어요!


from sklearn.metrics import mean_squared_error

# 학습한 데이터의 MSE(오류) 확인하기
train_mse = mean_squared_error(y_train, y_train_pred)
print("훈련데이터의 mse:",train_mse)
# 31.744431024367223

# 테스트 데이터의 MSE(오류) 확인하기
test_mse = mean_squared_error(y_test, y_pred)
print("테스트 데이터의 mse:",test_mse)

# 훈련데이터의 mse: 31.744431024367223
# 테스트 데이터의 mse: 26.798976627201053

#희귀모형에서도 교차검증을 할 수 있어요.
from sklearn.model_selection import cross_val_score

#교차검증에 사용 할 모델을 생성합니다















