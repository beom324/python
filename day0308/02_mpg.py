import pandas as pd
import numpy as np
df = pd.read_csv("../Data/mpg.csv")
'''
연습) mpg데이터를 읽어 들여 다음을 실습 해 봅니다.
연료별                 가격
c:CNC                 2.35
d:diesel              2.38
e:ethanol E85         2.11
p:premium             2.76
r:regular             2.11

1)위의 내용에 다른 연료 종류별 가격의 정보를 데이터프레임으로 만들어 봅니다.
2)mpg데이터에는 연료 종류를 나타내는 fl변수는 있지만 연료 가격을 나타내는 변수는 없습니다
  앞에서 만든 데이터를 이용하여 mpg데이터에 연료가격(price_fl)변수를 추가합니다.
3)연료가격 변수가 잘 추가가 되었는지 확인하기 위하여 model, fl, price_fl변수를 추출해 앞부분 5행을 출력해봅니다
'''
mpg = pd.DataFrame({
    'fl':['c','d','e','p','r'],
    'price_fl':[2.35,2.38,2.11,2.76,2.11]
})

mpg = mpg.merge(df)

print(mpg[['model','fl','price_fl']].head(30))
