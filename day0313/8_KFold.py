import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np

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



#학습한 평가를 차례로 저장하기 위한 리스트
val_score=[]
#훈련(평가)한 횟수를 증가시키기 위한 변수
num_fold=1

#교차검증을 위하여 학습데이터를  훈련데이터와 검증데이터로 분리하여
#인덱스를 알려주는 모델을 읽어 들입니다.
from sklearn.model_selection import KFold


# n_splits=5  전체데이터를 5개로 분할 , 교차검증을 5번
kfold = KFold(n_splits=5,shuffle=True,random_state=1234)


for tr_idx, val_idx in kfold.split(x_train,y_train):
    # 훈련용 문제, 검증용 문제
    x_tr,x_val=x_train.iloc[tr_idx,:],x_train.iloc[val_idx,:]
    #훈련용 답, 검증용 답
    y_tr,y_val= y_train.iloc[tr_idx],y_train.iloc[val_idx]

    #학습
    model = RandomForestClassifier(max_depth=5,random_state=20)
    model.fit(x_tr,y_tr)

    #검증하기
    y_val_pred = model.predict(x_val)

    #정확도 확인하기
    val_acc = accuracy_score(y_val,y_val_pred)

    print("%d 번쨰의 교차검증의 정확도 : %.4f "%(num_fold,val_acc))

    #전체 정확도를 담기 위한 배열에 추가한다.
    val_score.append(val_acc)

    #몇번째 학습(검증)했는지 변수를 1 증가 시킨다.
    num_fold +=1

#교차검증의 정확도 전체 평균 계산하여 출력
mean_score = np.mean(val_score)
print("평균 검증 정확도 : ",mean_score)
