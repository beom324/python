#매개변수 꽃잎길이, 꽃잎넓이, 꽃받침길이, 꽃받침넓이를 매개변수로 전달받아
#붓꽃의 품좀을 반환하도록 하는 함수를 만들어 봅니다
import pandas as pd
import numpy as np
import joblib
#학습한 모델을 저장하고 읽어들이기 위한 패키지 (joblib)
#저장 : dump
#읽기 : load

# dic={0:'setosa',1:'versicolor',2:'virginica'}
def getIris(sepal_length,sepal_width,petal_length,petal_width):
    result='setosa'
    knn=joblib.load("../Data/iris.pkl")
    x=pd.DataFrame({
        'sepal_length':[sepal_length],
        'sepal_width':[sepal_width],
        'petal_length':[petal_length],
        'petal_width':[petal_width]
    })
    y=knn.predict(x)
    y=y[0]

    if y==0:
        result = 'setosa'
    elif y==1:
        result ='versicolor'
    elif y==2:
        result ='virginica'
    else:
        print("잘못된 입력값입니다")

    return result
#dic={0:'setosa',1:'versicolor',2:'virginica'}

