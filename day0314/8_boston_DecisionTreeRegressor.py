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

#의사결정나무 알고리즘으로 회귀모형을 구현 ==> sklearn.tree.DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

#의사결정나무 알고리즘의 회귀모델 생성
model = DecisionTreeRegressor(max_depth=3,random_state=12)

#학습시키기
model.fit(X_train,y_train)

#공부한 데이터 예측시키기
y_train_pred = model.predict(X_train)
#공부안한 데이터 예측시키기
y_test_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

#공부한 데이터의 오류
train_mse=mean_squared_error(y_train,y_train_pred)
#공부안한 데이터의 오류
test_mse=mean_squared_error(y_test,y_test_pred)


print("공부한 데이터의 오류 : " , train_mse)
print("공부안한 데이터의 오류 : " , test_mse)
# 공부한 데이터의 오류 :  17.27751483586021
# 공부안한 데이터의 오류 :  24.983207329388055










