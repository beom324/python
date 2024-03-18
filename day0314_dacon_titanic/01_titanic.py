import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
submission = pd.read_csv("Data/submission.csv")

#(891, 12) (418, 11) (418, 2)
print(train.shape,test.shape,submission.shape)


#PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
# 0            1         0       3  ...   7.2500   NaN         S
# 1            2         1       1  ...  71.2833   C85         C
# 2            3         1       3  ...   7.9250   NaN         S
# 3            4         1       1  ...  53.1000  C123         S
# 4            5         0       3  ...   8.0500   NaN         S
print(train.head())



#       PassengerId  Pclass  ... Cabin Embarked
# 0          892       3  ...   NaN        Q
# 1          893       3  ...   NaN        S
# 2          894       2  ...   NaN        Q
# 3          895       3  ...   NaN        S
# 4          896       3  ...   NaN        S
# print(test.head())

#         PassengerId  Survived
# 0          892         0
# 1          893         1
# 2          894         0
# 3          895         0
# 4          896         1
# print(submission.head())

 #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64  <--
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object   <--
#  11  Embarked     889 non-null    object   <--
# print(train.info())

#각각의 속성(features)별로 요약 통계정보
print(train.describe())

#결측치 분포를 확인하기 위해 missingno 라이브러리를 사용
import missingno as msno
# msno.bar(train,figsize=(10,5),color=(0.7, 0.2, 0.2))
# plt.show()

#어느 위치에 결측치가 있는지 확인하는 함수 = missingno.matrix
# msno.matrix(train,figsize=(10,5),color=(0.7,0.7,0.2))
# plt.show()


#도화지크기
plt.figure(figsize=(8,8))
#글자크기
sns.set(font_scale=0.8)

#자료형이 문자인것은 제외하고 숫자인 피쳐들만 선택합니다.
# sel_col=['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare']
# train = train.loc[:,sel_col]

#상관계수테이블을 갖고 히트맵을 그려준다
# sns.heatmap(train.corr(),annot=True,cbar=True)
# plt.show()


#타이타닉 전체 데이터셋 준비
train['TrainSplit']="Train"
test['TrainSplit']='Test'

data = pd.concat([train,test], axis=0)

#분석에 필요한 숫자데이터만 추출
data_num = data.loc[:,['Pclass','Age','SibSp','Parch','Fare','Survived']]

print(data_num.info())
'''
 0   Pclass    1309 non-null   int64  
 1   Age       1046 non-null   float64  <---- 평균값(mean)로 대체
 2   SibSp     1309 non-null   int64  
 3   Parch     1309 non-null   int64  
 4   Fare      1308 non-null   float64   <----최빈값(mode)로 대체
 5   Survived  891 non-null    float64
'''

#결측치 처리하기
#나이에 대한 결측치는 평균값으로 변환하고 요금에 대한 결측치는 최빈값으로 변환하자.
# 0    8.05
#Name: Fare, dtype: float64
#최빈값을 갖고오면 key,value의 형태로 응답 값을 갖고 오기때문에
#최빈값을 가져오려면 [0]을 넣어줘야 value만 갖고와짐
data_num["Age"]=data_num["Age"].fillna(data_num['Age'].mean())
data_num["Fare"]=data_num["Fare"].fillna(data_num['Fare'].mode()[0])

print(data_num.info())
'''
 0   Pclass    1309 non-null   int64  
 1   Age       1309 non-null   float64
 2   SibSp     1309 non-null   int64  
 3   Parch     1309 non-null   int64  
 4   Fare      1309 non-null   float64
 5   Survived  891 non-null    float64
 결측치 제거됨
 '''

#목표변수 (Target - Survived)를 제외한 설명변수들을 정한다.
selected_features= ['Pclass','Age','SibSp','Parch','Fare']

#훈련데이터의 문제
X_train = data_num.loc[data["TrainSplit"]=='Train',selected_features]
#훈련데이터의 답
y_train = data_num.loc[data["TrainSplit"]=='Train','Survived']


#테스트데이터의 문제
X_test = data_num.loc[data["TrainSplit"]=='Test',selected_features]

# print(X_train.shape) #(891, 5)
# print(y_train.shape)#(891,)
# print(X_test.shape) #(418, 5)


#로지스틱 회귀모델을 이용하여 학습시켜봅시다
#훈련데이터와 검증데이터 분할
from sklearn.model_selection import  train_test_split
X_tr,X_val,y_tr,y_val=train_test_split(X_train,y_train,test_size=0.2,shuffle=True,random_state=1234)

#공부방법을 정해줍니다(로지스틱 회귀모델)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50,max_depth=3,random_state=20)
#공부시킨다
model.fit(X_tr,y_tr)

#예측한값
y_val_pred = model.predict(X_val)

    # #혼돈행렬(confusion Matrix)
    # from sklearn.metrics import confusion_matrix
    # #confusion Matrix를 갖고 heatmap을 그려줌 y_val = 실제 값 , y_val_pred = 예측 값
    # sns.heatmap(confusion_matrix(y_val,y_val_pred),annot=True)
    # plt.show()

#평가지표 출력하기               정확도            정밀도            재현율
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# print("정확도 : " , accuracy_score(y_val,y_val_pred))
# print("정밀도 : " , precision_score(y_val,y_val_pred))
# print("재현율 : " , recall_score(y_val,y_val_pred))
# print("f1 score : ", f1_score(y_val,y_val_pred))
#
# 정확도 :  0.7094972067039106
# 정밀도 :  0.7045454545454546
# 재현율 :  0.44285714285714284
# f1 score :  0.543859649122807

#dacon 에서 준 X_test를 이용해 예측하기
y_test_pred= model.predict(X_test)


submission['Survived'] = y_test_pred.astype(int)

#파일생성하기
submission.to_csv("./Data/titanic_004.csv",index=False)