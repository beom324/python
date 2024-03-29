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


@app.route("/gettitanic", methods=["GET"])
def getTitanic():
    result = 'no'
    pclass = request.args.get('pclass')
    sex = request.args.get('sex')
    age = request.args.get('age')
    sibsp = request.args.get('sibsp')
    parch = request.args.get('parch')
    fare = request.args.get('fare')
    embarked = request.args.get('embarked')
    who = request.args.get('who')

    result = isAlive(pclass,sex,age,sibsp,parch,fare,embarked,who)
    result = "titanic("+ str(result)+ ")"
    return result

if __name__ == '__main__':
    app.run()