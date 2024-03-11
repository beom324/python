import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import joblib

'''
 pclass   sex   age  sibsp  parch  fare embarked  who
0       3  male  22.0      1      0  7.25        S  man
==> 위와 같은 데이터를 
==> 학습한 모델에 적용하기 위해서는 
==> 다음과 같은 형태로 변환하여 예측 시킬수 있습니다.

     pclass   age  sibsp  parch  ...  embarked_S  who_child  who_man  who_woman
579       3  32.0      0      0  ...        True      False     True      False

'''


#  pclass       sex      age     sibsp   parch   fare    embarked    who
# 0       3     male     22.0      1      0     7.25        S       man


# 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_female', 'sex_male',
#        'embarked_C', 'embarked_Q', 'embarked_S', 'who_child', 'who_man',
#        'who_woman'

x = pd.DataFrame({
    'pclass':[0],
    'age':[22.0],
    'sibsp':[1],
    'parch':[0],
    'fare':[7.25],
    'sex_female':[0],
    'sex_male':[1],
    'embarked_C':[0],
    'embarked_Q':[0],
    'embarked_S':[1],
    'who_child':[0],
    'who_man':[1],
    'who_woman':[0]
})

# 학습모델 저장하고 읽어오기
# joblib.dump
# joblib.load

# 위의 데이터를 학습한 모델로 예측 시켜 봅시다.
model = joblib.load("../Data/titanic.pkl")
y = model.predict(x)
print(y)



