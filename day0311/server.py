
from flask import Flask
from flask import request
from flask import render_template
from myutil.titanic import isAlive
from flask import Flask, url_for
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route("/titanic",methods=['get'])
def insertForm():
    return render_template("titanic.html")


@app.route("/titanic",methods=['post'])
def insertSubmit():
    pclass = request.form['pclass']
    gender = request.form['gender']
    age = request.form['age']
    sibsp = request.form['sibsp']
    parch = request.form['parch']
    fare = request.form['fare']
    embarked = request.form['embarked']
    who = request.form['who']
    result = isAlive(pclass,gender,age,sibsp,parch,fare,embarked,who)

    return render_template("titanic.html", result=result)

if __name__ == '__main__':
    app.run()