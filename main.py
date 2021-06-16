from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from wtforms import Form, StringField, PasswordField, validators, SubmitField
import pickle
import pandas as pd
import numpy as np


sc = StandardScaler()
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    yazilar = {"Java", "Php", "Python", "Android", "Swift"}
    return render_template("index.html", yazilar=yazilar)


@app.route("/prediction", methods=["GET", "POST"])
def bayan():
    model = pickle.load(open("musto.hilalim", "rb"))

    income = request.form.get("income")
    age = request.form.get("age")
    rooms = request.form.get("rooms")
    bedrooms = request.form.get("bedrooms")
    population = request.form.get("population")

    bilgiler = {"income": [float(income)],
                "age": [float(age)],
                "rooms": [float(rooms)],
                "bedrooms": [float(bedrooms)],
                "population": [float(population)]
                }
    veri = pd.DataFrame.from_dict(bilgiler)
    
    values = [float(income),float(age),float(rooms),float(bedrooms),float(population)]
    values_array = np.array(values)
    values_array = values_array.reshape(-1, 1)
    values_sc = sc.fit_transform(veri)

    
    result = model.predict(values_sc)
    #print("tahmininiz")
    result = round(result[0])
    if request.method == "POST":
        return render_template("prediction.html", prediction = result)


if __name__ == '__main__':
    # DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=1512, debug=True)
