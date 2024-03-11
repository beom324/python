from flask import Flask
from flask import request
from myutil.titanic import isAlive

#terminal에서 직접 flask_cors install // pip install flask_cors
from flask_cors import CORS

app = Flask(__name__)
CORS(app) #외부의 다른 도메인으로부터 ajax통신을 허용한다.

# jsonp로 post방식의 요청은 되지 않는가? GET방식의 요청만 가능하다.
# jsonp로 응답은 list로만 가능한가 ? 문자열 데이터 하나는 홋따옴표나 쌍따옴표로 묶어서 응답한다.

@app.route("/gettitanic", methods=["POST"])
def getTitanic():
    data ="tiger"
    result = data
    return result

if __name__ == '__main__':
    app.run()