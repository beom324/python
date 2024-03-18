import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 피처 엔지니어링
# 각 피처(변수,속성)를 하나씩 모델 학습에 맞도록 전처리 하고
# 모델 성능 개선할 수 있는 방법을 찾는 것을 말한다.

train = pd.read_csv("./Data/train.csv")
test = pd.read_csv("./Data/test.csv")
submission = pd.read_csv("./Data/submission.csv")

df = pd.read_csv("./Data/df.csv")

# (['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'AgeBin',
#        'FamilySize', 'FareLog', 'Emb_Q', 'Emb_S', 'Cab_B', 'Cab_C', 'Cab_D',
#        'Cab_E', 'Cab_F', 'Cab_G', 'Cab_T', 'Cab_U', 'Tic_A4', 'Tic_A5',
#        'Tic_AQ3', 'Tic_AQ4', 'Tic_AS', 'Tic_C', 'Tic_CA', 'Tic_CASOTON',
#        'Tic_FC', 'Tic_FCC', 'Tic_Fa', 'Tic_LINE', 'Tic_LP', 'Tic_NUM',
#        'Tic_PC', 'Tic_PP', 'Tic_PPP', 'Tic_SC', 'Tic_SCA3', 'Tic_SCA4',
#        'Tic_SCAH', 'Tic_SCOW', 'Tic_SCPARIS', 'Tic_SCParis', 'Tic_SOC',
#        'Tic_SOP', 'Tic_SOPP', 'Tic_SOTONO2', 'Tic_SOTONOQ', 'Tic_SP',
#        'Tic_STONO', 'Tic_STONO2', 'Tic_STONOQ', 'Tic_SWPP', 'Tic_WC',
#        'Tic_WEP', 'Survived', 'TrainSplit']
print(df.columns)

select_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'AgeBin',
       'FamilySize', 'FareLog', 'Emb_Q', 'Emb_S', 'Cab_B', 'Cab_C', 'Cab_D',
       'Cab_E', 'Cab_F', 'Cab_G', 'Cab_T', 'Cab_U', 'Tic_A4', 'Tic_A5',
       'Tic_AQ3', 'Tic_AQ4', 'Tic_AS', 'Tic_C', 'Tic_CA', 'Tic_CASOTON',
       'Tic_FC', 'Tic_FCC', 'Tic_Fa', 'Tic_LINE', 'Tic_LP', 'Tic_NUM',
       'Tic_PC', 'Tic_PP', 'Tic_PPP', 'Tic_SC', 'Tic_SCA3', 'Tic_SCA4',
       'Tic_SCAH', 'Tic_SCOW', 'Tic_SCPARIS', 'Tic_SCParis', 'Tic_SOC',
       'Tic_SOP', 'Tic_SOPP', 'Tic_SOTONO2', 'Tic_SOTONOQ', 'Tic_SP',
       'Tic_STONO', 'Tic_STONO2', 'Tic_STONOQ', 'Tic_SWPP', 'Tic_WC',
       'Tic_WEP']

# 전체데이터로부터 훈련데이터의 답을 뽑아 옵니다.
y_tr = df.loc[df["TrainSplit"]=="Train","Survived"]
# 전체데이터로부터 훈련데이터의 문제를 뽑아 옵니다.
X_tr = df.loc[df["TrainSplit"]=="Train",select_features]
# 전체데이터로부터 테스트데이터의 문제를 뽑아 옵니다.
X_test = df.loc[df["TrainSplit"]=="Test",select_features]

#문제와 답안을 분류
#(891,) (891, 56)
#(418, 56)
# print(X_tr.shape,y_tr.shape)
# print(X_test.shape)

# 피쳐 중요도 속성을 갖고 있는 RandomForestClassifier로 학습 시켜 보도록 합시다.
# print(len(X_tr.columns)) #56

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1234)
model.fit(X_tr,y_tr)
y_test_predict=model.predict(X_test)
submission["Survived"]=y_test_predict.astype(int)

# submission.to_csv("./Data/titanic_day0318_02.csv",index=False)
# print("답안파일 작성")

print(model.feature_importances_)

a= np.array([2,1,10,6,3])
#np.argsort ==> 정렬했을때의 인덱스를 반환하는 함수,기본이 오름차순
#[10,6,3,2,1]
# 2, 3,4,0,1
#[1,2,3,6,10]
# 1,0,4,3,2
b=a.argsort() # 기본이 오름차순
#[1 0 4 3 2]
print(b)

# #피쳐 중요도를 그래프로 그려봅시다.
def plot_importance(model,features):
       #학습한 모델이 갖고있는 feature별 중요도를 뽑아서 저장한다 (56개)
       importances = model.feature_importances_

       #feature별 중요도를 갖고있는 배열을 오름차순 정렬했을때의 인덱스의 순서를 저장한다.
       indices = np.argsort(importances)

       #피쳐별 중요도가 있는 배열의 요소를 하나씩 꺼내와서 그 인덱스에
       #순서대로 피쳐이름에 담는다.
       feature_names = [features[i] for i in indices]

       #각 피쳐별 중요도가 있는 배열로부터 중요도 순서대로 갖고와서 배열을 만들어준다.
       feature_imp = importances[indices]

       #각 피쳐별 중요도를 가로 막대로 그려봅니다.
       plt.figure("Feature Importance")

       imp=feature_imp[:20]
       names = feature_names[:20]
       ind = indices[:20]

       #
       # plt.barh(range(len(ind)),imp,align="center")
       # plt.yticks(range(len(ind)),names)
       # plt.xlabel("Relative Importance")
       # plt.show()

       return list(reversed(feature_names)),list(reversed(feature_imp))


imp_features, imp_scores =plot_importance(model,select_features)

# 랜덤포레스트 모델이 중요하다고 판별한 10개의 속성만
# 추출하여 학습시켜 봅시다.

select_features = imp_features[:10]
y_train = df.loc[df["TrainSplit"]=="Train","Survived"]
X_train = df.loc[df["TrainSplit"]=="Train",select_features]
X_test = df.loc[df["TrainSplit"]=="Test",select_features]

# (891, 10) (891,)
# (418, 10)
print(X_train.shape,y_train.shape)
print(X_test.shape)

from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
model = XGBClassifier(max_depth=3,random_state=1234)
auc_scores =  cross_val_score(model,X_train,y_train,scoring="roc_auc")
print("개별 Fold AUC점수 : " , auc_scores)
print("평균 Fold AUC점수 : ", np.mean(auc_scores))

model.fit(X_train,y_train)
y_test_pred = model.predict(X_test)
submission["Survived"] = y_test_pred.astype(int)
submission.to_csv("./Data/titanic_day0318_04.csv",index=False)
print("답안파일 생성")