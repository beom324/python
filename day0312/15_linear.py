x=[-3,31,-11,4,0,22,-2,-5,-25,-14]
y=[-2,32,-10,5,1,23,-1,-4,-24,-13]

import matplotlib.pyplot as plt
# plt.plot(x,y)
# plt.show()

import pandas as pd
df= pd.DataFrame({
    'x':x,
    'y':y
})

train_features=['x']
target_cols=['y']
x_train=df.loc[:,train_features]
y_train=df.loc[:,target_cols]


#(10, 1)
#(10, 1)
# print(x_train.shape)#문제
# print(y_train.shape)#답

#모델학습
#LinearRegression(선형회귀 관계식,1차함수)   , y=ax+b
#LinearRegression은 데이터를 학습시키면 그 데이터를 가장 잘 설명할 수 있는 회귀식을 찾아줍니다.
#즉 y=ax+b 의 최적의 a(기울기)와 b(절편)를 찾아줍니다
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train) #fit = 공부를 시키는 명령어


#y=ax+b 의 최적의 a(기울기)와 b(절편)를 찾아줍니다
#기울기 = model.coef_
#절편 = model.intercept_
#[[1.]] [1.]
# print(model.coef_,model.intercept_)

#기울기 :  0.9999999999999999
#절편 :  0.9999999999999999
# print("기울기 : ",model.coef_[0][0])
# print("절편 : ",model.intercept_[0])

import numpy as np
x_new=np.array(11).reshape(1,1)
pred=model.predict(x_new)
##[[12.]]
# print(pred)


x_test=np.arange(11,16,1).reshape(-1,1)
y_pred= model.predict(x_test)
print(y_pred)



