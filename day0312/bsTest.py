import bs4
data = "<html><div>hello</div></html>"
bs_obj=bs4.BeautifulSoup(data,"html.parser")
# print(type(bs_obj))#<class 'bs4.BeautifulSoup'>
# print(bs_obj) #<html><div>hello</div></html>
a=bs_obj.find("div")
# print(a)       #<div>hello</div>
# print(type(a)) #<class 'bs4.element.Tag'>

