import bs4
data='''
<html>
    <ul class="ko">
        <li><a href="http://www.naver.com">네이버</a></li>
        <li><a href="http://www.daum.net">다음</a></li>
    </ul>
    <ul class="sns">
        <li><a href="http://www.google.com">구글</a></li>
        <li><a href="http://www.facebook.com">페이스북</a></li>
    </ul>
</html>
'''

bs_obj = bs4.BeautifulSoup(data,"html.parser")

# url =bs_obj.find('a')["href"]
# #http://www.naver.com
# print(url)

url = bs_obj.find_all('a')
for a in url:
    urls=a["href"]
    # http: // www.naver.com
    # http: // www.daum.net
    # http: // www.google.com
    # http: // www.facebook.com
    print(urls)


