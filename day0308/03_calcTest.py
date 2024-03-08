import calc as c
data = c.add(10,20)
print(data)
print(__name__)  #__main__
#실행을 한 모듈이면 __name__  =   __main__
#호출을 당한 곳에서는 module이름이 출력이됨
#__name__ calc