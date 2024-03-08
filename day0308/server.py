from flask import Flask
from flask import render_template
import memberDAO as dao

app = Flask(__name__) #__name__  = 모듈의 이름을 알려줌

@app.route("/hello")
@app.route("/hello/<irum>")
def hello(irum=None): #irum의 기본값은 None이다
    return render_template("hello.html",name=irum,age=20)#name이라는 변수명으로 irum을 상태유지해줌
    #render_templates시에는 templates에서 hello.html을 찾음


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/list')
def list():
    return render_template("listMember.html",list=dao.listMember())

from flask import request
@app.route("/insertMember",methods=['get'])
def insertMember():
    return render_template("insertMember.html")

@app.route("/insertMember",methods=['post'])
def insertSubmit():
    id=request.form["id"]
    name=request.form["name"]
    age=request.form["age"]
    doc={
        "id":id,
        "name":name,
        "age":age
    }
    dao.insertOne(doc)
    return render_template("listMember.html",list=dao.listMember())
@app.route("/insert/<id>/<name>/<age>")
def insert(id,name,age):
    dao.insertMember(id,name,age)
    return "ok"

if __name__ == '__main__': #여기가 출발점이라면 서버를 가동시켜라
    app.run()