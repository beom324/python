import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt
df=pd.DataFrame(iris['data'], columns=iris['feature_names'])
df.columns=['sepal_length','sepal_width','petal_length','petal_width']
df['Target']= iris['target']
df = df.drop_duplicates()
print(df.head())

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

from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(max_depth=3,random_state=20)

model.fit(x_train,y_train)#공부시킴 매개변수 = 문제(X_TRIAN),답(Y_TRAIN)을줌
pred = model.predict(x_test) #예측시킴 , 문제(X_TEST)만줌

# [0 1 1 2 1]
# 47     0
# 73     1
# 74     1
# 128    2
# 67     1
정확도 :  0.9333333333333333
print(pred[:5])
print(y_test[:5])

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred) #정답, 예측 값
#정확도 :  0.9333333333333333
print("정확도 : " , acc)

