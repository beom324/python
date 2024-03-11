import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import joblib

titanic = sns.load_dataset('titanic')
df = titanic.copy()
df = df.drop(columns=["class","adult_male","deck","embark_town","survived","alone"])
df = df.dropna()
target = df['alive']
df = df.drop(columns=['alive'])

print(df.iloc[5])
print(target.iloc[5])
print("-----------------------------------------")



#
# df = pd.get_dummies(df)
# df['alive']= target
#
# df_train, df_test = train_test_split(
#     df,
#     test_size=0.3,
#     random_state=1234,
#     stratify=df['alive']
# )
#
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(random_state=1234, max_depth=7)
# train_x = df_train.drop(columns=['alive'])
# train_y = df_train['alive']
#
# clf.fit(X=train_x, y=train_y)
#
# test_x = df_test.drop(columns=['alive'])
# test_y = df_test['alive']
#
# pred = clf.predict(test_x)
#
# acc= metrics.accuracy_score(y_true=test_y, y_pred=pred)
# prec = metrics.precision_score(y_true=test_y, y_pred=pred, pos_label='yes')
# recall = metrics.recall_score(y_true=test_y,y_pred=pred,pos_label='yes')
# f1 = metrics.f1_score(y_true=test_y,y_pred=pred,pos_label='yes')
#
# print(acc,prec,recall,f1)
#
# # 학습한 성능평가를 하여
# # 만족한 결과가 나왔다면 이것을
# # 어플리케이션에 활용하기 위하여 학습한 결과(모델)을 저장합니다.
# # 저장하기 위해서 joblib.dump함수를 이용하고 확장자는 pkl로 합니다.
# # 학습한 모델 저장하기
# joblib.dump(clf, "../Data/titanic.pkl")
# print('학습모델을 저장하였습니다.')
