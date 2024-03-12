import bs4
data = '''
<html>
    <body>
        <ul>
            <li>hello</li>
            <li>bye</li>
            <li>welcome</li>
        </ul>
    </body>
</html>
'''

bs_obj=bs4.BeautifulSoup(data,"html.parser")
ul=bs_obj.find("ul")
#<ul>
# <li>hello</li>
# <li>bye</li>
# <li>welcome</li>
#</ul>
# print(ul)

# 리스트형태로 반환
li_list=ul.find_all("li")
# [<li>hello</li>, <li>bye</li>, <li>welcome</li>]
# print(li_list)

# for li in li_list:
    # hello
    # bye
    # welcome
    # print(li.text)








