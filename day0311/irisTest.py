# import pandas as pd
# import numpy as np
# from sklearn import datasets
# import matplotlib.pyplot as plt
#
# iris = datasets.load_iris()
# print(type(iris))
# print(iris.keys())
# # <class 'sklearn.utils._bunch.Bunch'>
# # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
# print(iris['data'])
# print(iris['target'])
# print(iris['frame'])
# print(iris['target_names'])
# #['setosa' 'versicolor' 'virginica']
# #   0         1          2
# print(iris['DESCR'])
# print(iris['feature_names'])
# # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#
# '''
#     - sepal length in cm        : 꽃받침 길이
#     - sepal width in cm         : 꽃받침 넓이
#     - petal length in cm        : 꽃잎 길이
#     - petal width in cm         : 꽃잎 넓이
#     - class:
#             - Iris-Setosa
#             - Iris-Versicolour
#             - Iris-Virginica
# '''
# # [5.9   3.  5.1 1.8]   2
# print(iris['data'].shape)
# print(iris['target'].shape)
# # (150, 4)
# # (150,)
#
# df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
# print(df.head())
# print(df.shape)
# df.columns=['sepal_length','sepal_width','petal_length','petal_width']
# print(df.head())
#
# df['Target'] = iris['target']
# print(df.head())
# #    sepal_length  sepal_width  petal_length  petal_width  Target
# # 0           5.1          3.5           1.4          0.2       0
# # 1           4.9          3.0           1.4          0.2       0
# # 2           4.7          3.2           1.3          0.2       0
# # 3           4.6          3.1           1.5          0.2       0
# # 4           5.0          3.6           1.4          0.2       0
#
# print(df.tail())
# #      sepal_length  sepal_width  petal_length  petal_width  Target
# # 145           6.7          3.0           5.2          2.3       2
# # 146           6.3          2.5           5.0          1.9       2
# # 147           6.5          3.0           5.2          2.0       2
# # 148           6.2          3.4           5.4          2.3       2
# # 149           5.9          3.0           5.1          1.8       2
#
# # 탐색적 데이터분석
# # EDA (Exploratory Data Analysis)의 약어로 탐색적 데이터 분석을 말합니다.
# # 벨 연구소의 수학자 존튜키가 개발한 데이터분석 방법론으로
# # 데이터를 다양한 각도에서 관찰하고 이해하는 모든 과정을 말합니다.
# # 이과정을 통해서 학습에 불필요한 속성을 제거하고
# # 종속변수에 가장 크게 영향을 끼칠 속성들을 찾아 냅니다.
#
# print(df.info())
# # Data columns (total 5 columns):
# #  #   Column        Non-Null Count  Dtype
# # ---  ------        --------------  -----
# #  0   sepal_length  150 non-null    float64
# #  1   sepal_width   150 non-null    float64
# #  2   petal_length  150 non-null    float64
# #  3   petal_width   150 non-null    float64
# #  4   Target        150 non-null    int32
# # dtypes: float64(4), int32(1)
# # memory usage: 5.4 KB
# # None
#
# # 모든 속성이 숫자이므로 원핫인코딩의 대상이 없어요
# # 모든 속성이 결측치가 없으므로 결측치에 대한 처리를 따로 하지 않아도 됩니다.
#
# # 요약통계량 정보 확인하기
# print(df.describe())
# #        sepal_length  sepal_width  petal_length  petal_width      Target
# # count    150.000000   150.000000    150.000000   150.000000  150.000000
# # mean       5.843333     3.057333      3.758000     1.199333    1.000000
# # std        0.828066     0.435866      1.765298     0.762238    0.819232
# # min        4.300000     2.000000      1.000000     0.100000    0.000000
# # 25%        5.100000     2.800000      1.600000     0.300000    0.000000
# # 50%        5.800000     3.000000      4.350000     1.300000    1.000000
# # 75%        6.400000     3.300000      5.100000     1.800000    2.000000
# # max        7.900000     4.400000      6.900000     2.500000    2.000000
#
#
#
#
# # 결측치가 있는지 확인합니다.
# print(df.isna().sum())
# # sepal_length    0
# # sepal_width     0
# # petal_length    0
# # petal_width     0
# # Target          0
# # dtype: int64
#
# # 중복된 데이터가 있는지 확인
# print(df.duplicated().sum())
# # 1
#
# # 중복된 데이터 확인하기
# print(df.loc[df.duplicated(),:])
#
# #     sepal_length  sepal_width  petal_length  petal_width  Target
# # 142           5.8          2.7           5.1          1.9       2
#
# # 142번째 행과 동일한 데이터가 어디에 있는지 알아 봅시다.
# print(df.loc[(df.sepal_length == 5.8) & (df.petal_width == 1.9), :])
#
# #      sepal_length  sepal_width  petal_length  petal_width  Target
# # 101           5.8          2.7           5.1          1.9       2
# # 142           5.8          2.7           5.1          1.9       2
#
# # 중복된 데이터를 제거 합니다.
# df = df.drop_duplicates()
#
# print(df.shape)
# # (149, 5)
#
# # 아이리스의 품종에 가장 영향을 많이 끼지는 속성을 알아 보기 위하여 상관행렬을 출력 해 봅니다.
# print(df.corr())
#
# # 상관행렬표를 통하여 다음의 순서대로 Target에 영향을 끼지는 것을 알 수 있어요.
# # petal_width(0.956514) , petal_length(0.949402), sepal_length(0.786971))
# # 꽃잎넓이                 꽃잎길이                  꽃받침 길이
#
# # 상관행렬을 시각화하여 히트맵을 출력 해 봅시다.
# import seaborn as sns
# # sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# # plt.show()
#
# # 목표 레이블(Target)의 값의 종류별 빈도수를 확인
# print(df['Target'].value_counts())
# # Target
# # 0    50
# # 1    50
# # 2    49           <--- 중복된 데이터를 제거하여 하나가 줄었습니다.
#
# # petal_widthdml 히스토그램을 그려 봅니다.
# # plt.hist(x='petal_width',data=df)
# # plt.show()
#
# # seaborn의 displot함수를 이용하여 히스토그램을 그려 봅니다.
# # sns.displot(x='petal_width', kind='hist', data=df)
# # plt.show()
#
# # KDE(Kernel Density Estimation) : 커널밀도함수
# # 히스토그램을 부드럽게 곡선으로 나타냄
# # sns.displot(x='petal_width', kind='kde', data=df)
# # plt.show()
#
# # 품종(Target:0,1,2)별로 petal_width(꽃잎의 넓이) 분포를 확인
# # sns.displot(x='petal_width', kind='kde', data=df, hue="Target")
# # plt.show()
#
# # sns.displot(x='sepal_width', kind='kde', data=df, hue="Target")
# # plt.show()
#
# # 서로 다른 속성끼리  한꺼번에 상관관계 확인하기
# # sns.pairplot(df, hue="Target")
# # plt.show()
#
# from sklearn.model_selection import train_test_split
#
# # 데이터프레임으로 부터 문제와 답을 분리합니다.
# df.head(1)
# # sepal_length  sepal_width  petal_length  petal_width    Target
# X_data = df.loc[:, 'sepal_length':'petal_width']
# y_data = df.loc[:,'Target']
#
# print(X_data.shape)
# print(y_data.shape)
# # (149, 4)
# # (149,)
#
# # 데이터를 랜덤하게 섞고 훈련용, 테스트용 데이터로 분리합니다.
# X_train, X_test, y_train, y_test = train_test_split(X_data,
#                                                     y_data,
#                                                     test_size=0.2,
#                                                     shuffle=True,
#                                                     random_state=20)
#
# print(X_train.shape, y_train.shape)
# print(X_test.shape,  y_test.shape)
# # (119, 4) (119,)
# # (30, 4) (30,)
#
# # 어느 품종에 해당하는지 분류하기 위해서는 KNN을 이용합니다.
# # KNN(K-Nearest-Neibors)
# #       나와 가장 가까운 K개의 데이터를 찾아서
# #       그중에 가장 많이 포함된 품종으로 분류하는 알고리즘을 말한다.
#
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=7)
# model.fit(X_train, y_train)                 #<---- 훈련을 시켜요
#
# y_pred = model.predict(X_test)
# print("예측값:",y_pred[:5])
# print("실제값:",y_test)
#
# # 얼마나 공부를 잘 했는지 성능평가(정확도) 해 봅시다.
# from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_pred)
# print(score)
# # 0.9666666666666667
