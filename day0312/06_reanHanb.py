import urllib.request

import bs4

url="https://www.hanbit.co.kr/store/books/new_book_list.html"
html = urllib.request.urlopen(url)
data=html.read()
#url의 페이지소스보기에 소스코드를 가져옴
# print(data)

bs_obj= bs4.BeautifulSoup(data,"html.parser")

print(type(html))
print(type(data))

list = bs_obj.find_all("p",{"class":"book_tit"})

for a in list:
    print(a.text)













