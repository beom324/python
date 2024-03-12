data={"name":"kim","age":20,"grade":[3.0,4.0,3.5,4.2]}
try:
    print(data['name'])
    print(data['addr'])
except Exception as e:
    print("존재하지 않는 키입니다")
    print("예외발생 : " ,e)
finally:
    print("작업완료")