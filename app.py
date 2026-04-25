
from flask import Flask, render_template, request, redirect
import pandas as pd
import os
import shutil
from modules.preprocess import full_preprocess
from modules.train import train_all_models
from modules.predict import load_model_and_predict

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

# Ensure folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_FOLDER = "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

memory = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        if file.filename.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)

        memory["df"] = df

        return render_template("preview.html",
                               tables=[df.head().to_html()],
                               cols=df.columns)

    return render_template("index.html")

@app.route("/preprocess", methods=["POST"])
def preprocess():
    target = request.form["target"]
    df = memory["df"]

    X, y, meta = full_preprocess(df, target)
    memory["X"], memory["y"], memory["meta"] = X, y, meta

    return redirect("/train")

@app.route("/train")
def train():
    X, y = memory["X"], memory["y"]
    results, best = train_all_models(X, y)

    memory["best_model"] = best

    return render_template("train.html", results=results)


import numpy as np

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        meta = memory["meta"]
        features = meta["features"]
        encoders = meta["encoders"]
        scaler = meta["scaler"]

        values = []

        for f in features:
            val = request.form.get(f)

            # Handle categorical columns
            if f in encoders:
                le = encoders[f]
                try:
                    val = le.transform([val])[0]
                except:
                    val = 0   # fallback for unseen category
            else:
                val = float(val)

            values.append(val)

        
        values = scaler.transform([values])

        
        pred = memory["best_model"].predict(values)[0]

        return render_template("result.html", pred=pred)

    return render_template("predict.html", features=memory["meta"]["features"])

    # return render_template("predict.html")
    return render_template("predict.html", features=memory["meta"]["features"])

if __name__ == "__main__":
    app.run(debug=True)
