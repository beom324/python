import bs4
data = '''
<html>
    <body>
        <ul class='greet'>
            <li>hello</li>
            <li>bye</li>
            <li>welcome</li>
        </ul>
        <ul class='reply'>
            <li>ok</li>
            <li>no</li>
            <li>sure</li>
        </ul>
    </body>
</html>
'''
bs_obj=bs4.BeautifulSoup(data,"html.parser")
ul=bs_obj.find("ul",{"class":"reply"}) #class가 reply인 ul을 찾아라
# <ul class="reply">
# <li>ok</li>
# <li>no</li>
# <li>sure</li>
# </ul>
print(ul)






