import pandas as pd
import numpy as np
import joblib


#학습한 모델을 저장하고 읽어들이기 위한 패키지 (joblib)
#저장 : dump
#읽기 : load
knn=joblib.load("../Data/iris.pkl")

#꽃잎의 길이, 꽃잎의 넓이, 꽃받침 길이, 꽃받침 넓이가 필요
# x=np.array([5.7,3.0,4.2,1.2]).reshape(-1,4)
x=pd.DataFrame({
    'sepal_length':[5.7],
    'sepal_width':[3.0],
    'petal_length':[4.2],
    'petal_width':[1.2]
})
y=knn.predict(x)
