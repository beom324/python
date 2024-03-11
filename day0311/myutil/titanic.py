# 다음과 같은 매개변수를 전달받아 학습모델(../Data/titanic.pkl)로 부터
# 생존여부를 예측하여 반환하는 함수를 만들고 호출 해 봅니다.
# pclass       sex      age     sibsp   parch   fare    embarked    who

import joblib
import pandas as pd

def isAlive(pclass, sex, age, sibsp, parch, fare, embarked, who):
    y = 'no'
    model = joblib.load('../Data/titanic.pkl')
    x = pd.DataFrame({
        'pclass': [pclass],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'sex_female': [0],
        'sex_male': [0],
        'embarked_C': [0],
        'embarked_Q': [0],
        'embarked_S': [0],
        'who_child': [0],
        'who_man': [0],
        'who_woman': [0]
    })

    if sex == "male":
        x['sex_male'] = 1
    else:
        x["sex_female"] = 1

    if embarked == "C":
        x['embarked_C']  = 1
    elif embarked == "Q":
        x['embarked_Q'] = 1
    else:
        x['embarked_S'] = 1

    if who ==  "child":
        x['who_child'] = 1
    elif who == "man":
        x['who_man'] = 1
    else:
        x['who_woman']= 1

    y = model.predict(x)

    return y[0]






