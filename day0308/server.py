
from flask import Flask
from flask import render_template
import memberDAO as dao
from flask import Flask, url_for
app = Flask(__name__)

from flask import request
from myUtil.sist import makeWordCloud

@app.route("/wc", methods=["GET"])
def wcForm():
    return render_template("wc.html")

# fname, stop_word, imgName, font
@app.route("/wc", methods=["POST"])
def wcSubmit():
    fname = request.form["fname"]
    stop_word = request.form["stop_word"]
    imgName = request.form["imgName"]
    font = request.form["font"]
    data = stop_word.split(",")
    makeWordCloud(fname,data,imgName,font)
    return render_template("wc_ok.html", imgName=imgName)




@app.route("/insertMember", methods=["GET"])
def insertMember():
    return render_template("insertMember.html")

@app.route("/insertMember", methods=["POST"])
def insertSubmit():
    id = request.form["id"]
    name = request.form["name"]
    age = request.form["age"]
    doc = { "id":id,   "name":name,  "age":age  }
    dao.insertMember(doc)
    return render_template("listMember.html", list=dao.listMember())


# listMember라고 요청하면
# 모든 고객목록을 출력하는 웹페이지를 구현 해 봅니다.
@app.route("/listMember")
def listMember():
    data= list(dao.listMember())
    for row in data:
        del(row['_id'])


    print(data)
    print(type(data))
    return "pro("+str(data)+")" #함수의 형태로 리턴해줘야함

@app.route("/hello/")
@app.route("/hello/<irum>")
def hello(irum=None):
    return render_template("hello.html",name=irum,age=20)

@app.route('/')
def hello_world():
    return 'Hello World!'

# @app.route("/list")
# def list():
#     return "List Book"



if __name__ == '__main__':
    app.run()
