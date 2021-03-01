from flask import Flask, redirect, url_for, render_template, request
import numpy as np
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("user", usr=user))
    else:
        return render_template("input.html")

@app.route("/<usr>")
def user(usr):
    usr = np.random.uniform(0.01, 1.0)
    return f"<h1>{usr}</h1>"

if __name__ == "__main__":
    app.run(debug=True)