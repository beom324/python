import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt
df=pd.DataFrame(iris['data'], columns=iris['feature_names'])
df.columns=['sepal_length','sepal_width','petal_length','petal_width']
df['Target']= iris['target']
df = df.drop_duplicates()
# print(df.head())

from sklearn.model_selection import train_test_split #훈련데이터와 테스트데이터를 분리해주는 함수


x_data=df.loc[:,'sepal_length':'petal_width']  #모든행에(sepal_length부터 petal_length 까지) 데이터를 갖고와라
y_data = df.loc[:,'Target']
x_train,x_test,y_train,y_test=train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    shuffle=True,
    random_state=20
)
#전체 데이터로부터 학습데이터와 테스트데이터로 분리 한 상태입니다.
#학습데이터   : x_train, y_train
#테스트데이터  : x_test, y_test

#위의 분리된 학습데이터(x_train,y_train)를 교차검증을 위해 다시 훈련데이터와 검증데이터로 분리해보자.

#학습데이터의 shape
#(119,4)
#(119, )
# print(x_train.shape)
# print(y_train.shape)
#학습데이터 문제와 답을 갖고 훈련데이터와 검증데이터로 분리합니다.
x_tr,x_val,y_tr,y_val=train_test_split(x_train,y_train,test_size=0.3,shuffle=True,random_state=20)



# (83, 4) (83,)
# (36, 4) (36,)
# print(x_tr.shape,y_tr.shape)
# print(x_val.shape,y_val.shape)

from sklearn.ensemble import RandomForestClassifier

#n_estimators 생략시 기본이 100개
model = RandomForestClassifier(max_depth=3,random_state=20)

#학습
model.fit(x_tr,y_tr)

#공부한 데이터를 예측시키기
y_tr_predict = model.predict(x_tr)
#검증데이터 예측
y_val_predict = model.predict(x_val)

#성능평가
from sklearn.metrics import accuracy_score

tr_acc = accuracy_score(y_tr,y_tr_predict)
val_acc = accuracy_score(y_val,y_val_predict)

print("공부한것의 성능 : " ,tr_acc)
print("검증데이터에 대한 성능 : " , val_acc)

#테스트 데이터로 예측시키고 성능 확인
y_test_pred =model.predict(x_test)
test_acc = accuracy_score(y_test,y_test_pred)

print("테스트데이터에 대한 성능 ", test_acc)

# 공부한것의 성능 :  0.9879518072289156
# 검증데이터에 대한 성능 :  0.9166666666666666
# 테스트데이터에 대한 성능  0.9

# 과적합(overfit) : 공부한것만 잘 알아 맞추고 새로운 데이터에 대해서는 예측률이 낮아요.
