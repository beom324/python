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

#둘중에 하나 선택하는 공부방법                   = 의사결정나무(DecisionTreeClassifier)
#여러개중에 어디에 해당하는지 분류하는 공부방법     =K-최근접이웃(KNN,K-Nearest Neighbors)    ,k=데이터갯수
# ---> 나와 가장 가까운 k를 뽑아서 그중에 가장 많은 빈도에 속하도록 하는 방법

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)#공부시킴 매개변수 = 문제(X_TRIAN),답(Y_TRAIN)을줌
pred = knn.predict(x_test) #예측시킴 , 문제(X_TEST)만줌


# from sklearn.metrics import accuracy_score
# knn_score = accuracy_score(y_test,pred)
# print(knn_score)

#학습한 모델을 저장하고 읽어들이기 위한 패키지
#저장 : dump
#읽기 : load
import joblib
joblib.dump(knn,"../Data/iris.pkl")
print("모델을 저장하였습니다")