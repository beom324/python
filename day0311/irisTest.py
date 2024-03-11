import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris() #iris데이터를 읽어주는 함수
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
# print(iris.keys())
# print(iris['data']) #문제
# print(iris['target']) #답
# print(iris['target_names'])
# print(iris['DESCR'])
'''
:Attribute Information:
    - sepal length in cm     :꽃 받침의 길이
    - sepal width in cm       :꽃 받침의 넓이
    - petal length in cm     :꽃잎의 길이
    - petal width in cm      :꽃잎의 넓이
    - class:
            - Iris-Setosa
            - Iris-Versicolour
            - Iris-Virginica
'''
# print(iris['data'].shape)
# print(iris['feature_names'])

df = pd.DataFrame(iris['data'],columns=iris['feature_names'])
df.columns=['sepal_length','sepal_width','petal_length','petal_width']
# print(df.head())
# print(df.shape)

#탐색적 데이터분석
#EDA(Exploratory Data Analysis)의 약어로 탐색적 데이터 분석을 말합니다.
#벨 연구소의 수학자 존튜키가 개발한 데이터 분석 방법론으로
#데이터를 다양한 각도에서 관찰하고 이해하는 모든 과정을 말합니다
#이 과정을 통해서 학습에 불필요한 속성을 제거하고
#종속변수에 가장 크게 영향을 끼칠 속성들을 찾아 냅니다.

print(df.info())

#모든 속성이 숫자이므로 원핫인코딩의 대상이 없어요
#모든 속성이 결측치가 없으므로 결츨치에 대한 처리를 따로 하지 않아도 됩니다.

#요약통계량 정보 확인하기
print(df.describe())
#결측치 확인하기
print(df.isna().sum())
#중복된 데이터가 있는지 확인(1개 존재)
print(df.duplicated().sum())

#중복된 데이터 확인하기
print(df.loc[df.duplicated(),:])
#     sepal_length  sepal_width  petal_length  petal_width
#142           5.8          2.7           5.1          1.9

#142번째 행과 동일한 데이터가 어디에 있는지 알아 봅시다.
print(df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9)])
#     sepal_length  sepal_width  petal_length  petal_width
#101           5.8          2.7           5.1          1.9
#142           5.8          2.7           5.1          1.9



#중복된 데이터를 제거합니다.
df = df.drop_duplicates();

#iris의 품종에 가장 영향을 많이 끼치는 속성을 알아 보기 위하여 상관행렬을 출력 해 봅니다.
print(df.corr())
'''
              sepal_length  sepal_width  petal_length  petal_width
sepal_length      1.000000    -0.118129      0.873738     0.820620
sepal_width      -0.118129     1.000000     -0.426028    -0.362894
petal_length      0.873738    -0.426028      1.000000     0.962772
petal_width       0.820620    -0.362894      0.962772     1.000000

'''

# #히트맵을 출력 해 봅시다
import seaborn as sns
# sns.heatmap(data=df.corr(),square=True,annot=True,cbar=True)
# plt.show()

# #목표 레이블(Target)의 값의 종류별 빈도수를 확인
# print(df['Target'].value_counts())

# #petal_width의 히스토그램을 그려 봅니다.
# plt.hist(x='petal_width',data=df)
# plt.show()

#seaborn의 displot함수를 이용해서 히스토그램을 그려 봅니다.
# sns.displot(x='petal_width',kind='hist',data=df)
# plt.show()

#KDE(Kernel Destiny Estimation) : 커널밀도함수
#히스토그램을 부드럽게 곡선으로 나타냄
# sns.displot(x='petal_width',kde='hist',data=df)
# plt.show()


#품종 (Target:0,1,2)별로 petal_width(꽃잎의 넓이) 분포를 확인
# sns.displot(x='petal_width',kde='hist',data=df, hue='Target')
# plt.show()











