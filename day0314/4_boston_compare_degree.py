import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# data를 갖고 데이터프레임을 만들어요
df = pd.DataFrame(data)
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

# 데이터프레임에 target을 설정합니다.
df_target = pd.DataFrame(target)
df_target.columns = ["Target"]

df = pd.concat([df, df_target],axis=1)

df_corr= df.corr()
import seaborn as sns

corr_order = df.corr().loc[:'LSTAT', 'Target'].abs().sort_values(ascending=False)

# 타겟(집값)에 가장 영향을 많이 끼치는 4개의 피쳐(속성)만 뽑아오도록 합시다.
plot_cols = ['Target','LSTAT','RM','PTRATIO','INDUS']
plot_df = df.loc[:, plot_cols]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 타겟변수(집값)을 제외한 속성들을 뽑아 옵니다.
df_scaler = df.iloc[:,:-1]


# 스케일할 준비를 시켜요.
scaler.fit(df_scaler)

# 준비한 대로 스케일링을 적용시켜
df_scaler = scaler.transform(df_scaler)

df.iloc[:,:-1] = df_scaler[:,:]

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

# 학습문제, 테스트문제, 학습답, 테스트답으로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(
                                            X_data, y_data,
                                            test_size=0.2, random_state=1234, shuffle=True)

# 과대적합(overfitting)
# ==>훈련데이터와 비슷한 데이터는 잘 예측하지만 새로운 데이터에 예측력이 떨어지는 현상을 말합니다.
#    이상치까지 너무 세세하게 학습하다 보니 이런 현상이 있을 수 있어요.
# 과소적합(underfitting)
# ==>훈련데이터의 특성이 덜 파악된 형태를 말합니다.
#    데이터 양이 부족한 경우에 발생 할 수 있어요.

# 과적합의 문제를 해소하기 위하여 '단항식'이 아닌 '다항식'의 선형회귀를 이용할 수 있어요.
# 문제(features)들을 다항식에 맞도록 변환해주는 함수 = sklearn.preprocessing.PolynomialFeatures


#훈련데이터를 다항식(2차항)에 맞도록 변경 해 봅시다.
print(X_train.head())
#         LSTAT        RM
# 221  0.544426  0.498755
# 375  0.323124  0.718912
# 450  0.433499  0.610845
# 163  0.043874  0.922399
# 402  0.512693  0.544740

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)


x_train_poly=pf.fit_transform(X_train)
# [[1.         0.54442605 0.49875455 0.29639972 0.27153497 0.2487561 ]
#  [1.         0.32312362 0.71891167 0.10440887 0.23229734 0.51683399]
#  [1.         0.4334989  0.61084499 0.18792129 0.26480063 0.3731316 ]
#  [1.         0.04387417 0.92239893 0.00192494 0.04046949 0.85081978]
#  [1.         0.51269316 0.54474037 0.26285427 0.27928466 0.29674207]]
print(x_train_poly[:5,:])

#다항식(2차항)으로 변환된 문제를 갖고 공부시키도록 합니다.
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train_poly,y_train)

#테스트 데이터도 공부한 2차항으로 변환후에 예측을 시켜야 합니다
x_test_poly = pf.fit_transform(X_test)

#공부한 데이터 예측
y_train_pred = model.predict(x_train_poly)
#공부 안한 데이터 예측
y_test_pred = model.predict(x_test_poly)

from sklearn.metrics import mean_squared_error

#공부한 데이터의 오류(mse)
train_mse = mean_squared_error(y_train,y_train_pred)
#공부안한 데이터의 오류(mse)
test_mse = mean_squared_error(y_test,y_test_pred)


# train_mse 20.87181490592505
# test_mse  19.457145102295346
print("train_mse" , train_mse)
print("test_mse ", test_mse)

#1차항일때의 mse
# 훈련데이터의 mse: 31.744431024367223
# 테스트 데이터의 mse: 26.798976627201053

#2차항일때의 mse
# train_mse 20.87181490592505
# test_mse  19.457145102295346

#1차항 모델보다 2차항 모델이 성능이 더 좋은편이다.
#그럼 차수가 높을 수록 좋은가?
#15차항으로 실험
# train_mse 56.040272277227714
# test_mse  71869421523.73256
#차수를 너무 높인다고 좋은것이 아님.

#차수가 1차,2차,15차일떄의 실제 데이터와 예측데이터를 산점도를 찍어서 비교 해봅시다.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

plt.figure(figsize=(15,5))
for n, deg in enumerate([1,2,15]):
    ax1 = plt.subplot(1,3,n+1)
    pf=PolynomialFeatures(degree=deg)
    x_train_poly = pf.fit_transform(X_train.loc[:,["LSTAT"]])
    x_test_poly = pf.fit_transform(X_test.loc[:,["LSTAT"]])
    model=LinearRegression()

    model.fit(x_train_poly,y_train)
    y_test_pred = model.predict(x_test_poly)

    #실제값 그래프 그리기
    plt.scatter(X_test.loc[:,"LSTAT"],y_test,label="Target")

    #예측값 그래프 그리기
    plt.scatter(X_test.loc[:,"LSTAT"],y_test_pred,label="Pred")

    #plt의 제목 설정하기
    plt.title("Degree %d"%deg)

    #범례표시
    plt.legend()


plt.show()


