from flask import request
from flask import render_template
from flask import Flask, url_for
from iris import getIris
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/iris",methods=['get'])
def irisForm():
    return render_template("/iris.html")

@app.route("/iris",methods=['post'])
def irisSubmit():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])
    result = getIris(sepal_length,sepal_width,petal_length,petal_width)

    return render_template("/iris.html",results=result)

@app.route("/getIris",methods=['post'])
def irisSubmitAjax():
    sepal_length = float(request.form["sepal_length"])
    sepal_width = float(request.form["sepal_width"])
    petal_length = float(request.form["petal_length"])
    petal_width = float(request.form["petal_width"])
    result = getIris(sepal_length,sepal_width,petal_length,petal_width)

    return result

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run()