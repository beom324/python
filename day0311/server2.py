from flask import Flask
from flask import request
from flask import render_template
from myutil.titanic import isAlive
from flask import Flask, url_for
app = Flask(__name__)

# jsonp로 post방식의 요청은 되지 않는가? GET방식의 요청만 가능하다.
# jsonp로 응답은 list로만 가능한가 ? 문자열 데이터 하나는 홋따옴표나 쌍따옴표로 묶어서 응답한다.
# jsonp로 요청하니까? 응답도 json으로 응답하는 것이 마땅


@app.route("/gettitanic", methods=["GET"])
def getTitanic():
    data ="tiger"
    result = "pro('"+data+"')"
    return result


if __name__ == '__main__':
    app.run()