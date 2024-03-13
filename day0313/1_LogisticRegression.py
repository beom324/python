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


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
'''
SVC Kernel의 종류
1.선형 (linear)
2.다항식 커널 (poly)
3.가우시안 커널 (rbf)   :  대체로 성능이 가장 우수,가장 많이 쓰이는 기법
4.시그모이드 커널 (sigmoid)
'''

model.fit(x_train,y_train)#공부시킴 매개변수 = 문제(X_TRIAN),답(Y_TRAIN)을줌
pred = model.predict(x_test) #예측시킴 , 문제(X_TEST)만줌

#각 품종별(클래스 별) 확률값을 출력 해 봅시다
# --> model.predict_propa(각 품종별 확률값을 알려줌)
# --> model.predict(결과예측)
y_pred = model.predict(x_test)
y_pred_prob = model.predict_proba(x_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,pred) #정답, 예측 값
# print("정확도 : " , acc)
print(y_pred[:5])
print(y_pred_prob[:5])
# [0 1 1 2 1]
#      0일확률         1일확률           2일확률
# [[9.83139538e-01 1.68604045e-02 5.74324447e-08]
#  [4.60555853e-03 8.41674340e-01 1.53720101e-01]
#  [1.03262949e-02 9.20317508e-01 6.93561974e-02]
#  [2.57738050e-05 5.16141924e-02 9.48360034e-01]
#  [2.39265716e-02 9.52076335e-01 2.39970929e-02]]

