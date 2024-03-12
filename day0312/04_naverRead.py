import urllib.request

import bs4

url="https://www.naver.com"
html = urllib.request.urlopen(url)
data=html.read()
#url의 페이지소스보기에 소스코드를 가져옴
# print(data)

bs_obj= bs4.BeautifulSoup(data,"html.parser")

# <class 'bs4.BeautifulSoup'>
# <class 'bytes'>
# print(type(bs_obj))
# print(type(data))
# print(bs_obj)
# #원하는 노드 찾는방법
a=bs_obj.find("div",{"id":"shortcutArea"})#bs
print(a)








