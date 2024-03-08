from flask import Flask
from flask import render_template
app = Flask(__name__) #__name__  = 모듈의 이름을 알려줌

@app.route("/hello")
@app.route("/hello/<irum>")
def hello(irum=None): #irum의 기본값은 None이다
    return render_template("hello.html",name=irum)#name이라는 변수명으로 irum을 상태유지해줌
    #render_templates시에는 templates에서 hello.html을 찾음



@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/list')
def list():
    return "List Book"

if __name__ == '__main__': #여기가 출발점이라면 서버를 가동시켜라
    app.run()