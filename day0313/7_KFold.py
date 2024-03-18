import pandas as pd
import numpy as np
from sklearn import datasets
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


#n_splits=5  전체데이터를 5개로 분할 , 교차검증을 5번
kfold = KFold(n_splits=5,shuffle=True,random_state=1234)


for tr_idx, val_idx in kfold.split(x_train,y_train):
    print("훈련용 인덱스  : ",tr_idx)
    print("검증용 인덱스 : " , val_idx)

    # 훈련용
    # 인덱스: [0   2   3   4   5   7  10  11  12  13  14  15  16  17  18  19  20  21
    #       23  24  25  26  27  28  30  31  33  34  35  36  37  38  39  40  41  43
    #       44  45  46  47  48  49  50  51  53  55  56  57  58  59  60  61  62  63
    #       65  67  68  69  71  73  75  76  77  78  79  80  81  83  84  87  88  89
    #       90  92  93  94  95  96  98 100 101 102 103 104 105 106 107 108 110 111
    #       112 114 116 117 118]
    # 검증용
    # 인덱스: [1   6   8   9  22  29  32  42  52  54  64  66  70  72  74  82  85  86
    #       91  97  99 109 113 115]
    # 훈련용
    # 인덱스: [0   1   2   3   5   6   7   8   9  11  12  13  14  15  16  18  19  21
    #       22  23  24  26  28  29  30  31  32  34  36  37  38  41  42  43  44  45
    #       46  47  49  50  51  52  53  54  56  58  60  61  64  65  66  67  68  69
    #       70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87
    #       89  91  92  93  94  95  97  98  99 100 101 105 107 108 109 111 112 113
    #       114 115 116 117 118]
    # 검증용
    # 인덱스: [4  10  17  20  25  27  33  35  39  40  48  55  57  59  62  63  88  90
    #       96 102 103 104 106 110]
    # 훈련용
    # 인덱스: [0   1   2   3   4   6   8   9  10  11  12  13  14  15  17  19  20  22
    #       23  24  25  26  27  28  29  30  31  32  33  34  35  37  38  39  40  42
    #       43  46  47  48  49  50  52  53  54  55  57  58  59  60  62  63  64  65
    #       66  67  69  70  71  72  73  74  75  76  80  81  82  83  84  85  86  88
    #       89  90  91  92  96  97  99 100 101 102 103 104 105 106 107 108 109 110
    #       113 114 115 116 118]
    # 검증용
    # 인덱스: [5   7  16  18  21  36  41  44  45  51  56  61  68  77  78  79  87  93
    #       94  95  98 111 112 117]
    # 훈련용
    # 인덱스: [1   4   5   6   7   8   9  10  15  16  17  18  20  21  22  23  24  25
    #       26  27  29  30  32  33  35  36  38  39  40  41  42  43  44  45  47  48
    #       49  50  51  52  53  54  55  56  57  58  59  61  62  63  64  66  68  69
    #       70  72  73  74  76  77  78  79  80  82  83  85  86  87  88  90  91  92
    #       93  94  95  96  97  98  99 102 103 104 105 106 107 108 109 110 111 112
    #       113 114 115 117 118]
    # 검증용
    # 인덱스: [0   2   3  11  12  13  14  19  28  31  34  37  46  60  65  67  71  75
    #       81  84  89 100 101 116]
    # 훈련용
    # 인덱스: [0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  16  17  18
    #       19  20  21  22  25  27  28  29  31  32  33  34  35  36  37  39  40  41
    #       42  44  45  46  48  51  52  54  55  56  57  59  60  61  62  63  64  65
    #       66  67  68  70  71  72  74  75  77  78  79  81  82  84  85  86  87  88
    #       89  90  91  93  94  95  96  97  98  99 100 101 102 103 104 106 109 110
    #       111 112 113 115 116 117]
    # 검증용
    # 인덱스: [15  23  24  26  30  38  43  47  49  50  53  58  69  73  76  80  83  92
    #       105 107 108 114 118]