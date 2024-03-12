import pandas as pd
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import matplotlib.pyplot as plt
# print(iris)
# print(type(iris)) #보통 확인해보면 type = dataFrame인데 얘는 Bunch라는 타입.
                    #Bunch는 sklearn패키지가 만들어 놓은 딕셔너리와 유사한 자료형

# print(iris.keys())

#문제
# print(iris['data'])

#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris['feature_names'])
#답
# print(iris['target'])

#['setosa' 'versicolor' 'virginica']
# print(iris['target_names'])


# print(iris['DESCR'])

# print(iris['data'].shape)#(150,4)
# print(iris['target'].shape)#(150,)

df=pd.DataFrame(iris['data'], columns=iris['feature_names'])
# print(df.head())
# print(df.shape)

df.columns=['sepal_length','sepal_width','petal_length','petal_width']
# print(df.head())
df['Target']= iris['target']


#모든변수에 결측치가 없다.
#모든변수가 숫자 자료형
# print(df.info())


#중복된 데이터 제거하기(1행 제거됨)
df = df.drop_duplicates()

#어떤것이 가장 영향을 많이 끼칠까?
#==>상관행렬표, 상관히트맵 만들어서 확인해보기
#상관행렬표 = print(df.corr())
#petal_length가 iris의 품종을 결정하는데 가장 중요한 요인
#               sepal_length  sepal_width  petal_length  petal_width    Target
# sepal_length      1.000000    -0.118129      0.873738     0.820620  0.786971
# sepal_width      -0.118129     1.000000     -0.426028    -0.362894 -0.422987
# petal_length      0.873738    -0.426028      1.000000     0.962772  0.949402
# petal_width       0.820620    -0.362894      0.962772     1.000000  0.956514
# Target            0.786971    -0.422987      0.949402     0.956514  1.000000
#히트맵을 그리기
import seaborn as sns
# sns.heatmap(df.corr(),annot=True) #annot=True는 값도 출력
# plt.show()

#Target(품종)의 종류별 빈도수
# 0    50
# 1    50
# 2    49
# print(df['Target'].value_counts())

#속성 pental_width의 히스토그램 그리기
# plt.hist(x='petal_width',data=df)
# plt.show()

#속성 pental_width의 히스토그램 그리기(seaborn이용)
# sns.histplot(data=df,kde='kind', x='petal_width',hue='Target')
# plt.show()


#모든 속성에 대하여 속성별로 상관관계 확인
# sns.pairplot(data=df,hue='Target')
# plt.show()

