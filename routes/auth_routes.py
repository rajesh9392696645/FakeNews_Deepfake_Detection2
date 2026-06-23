from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from database.db_operations import register_user, login_user

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/")
def home():
    return redirect(url_for("auth.login"))

@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = login_user(email, password)

        if user:
            session["user_id"] = user[0]
            session["username"] = user[1]

            return redirect(url_for("dashboard.dashboard"))

        flash("Invalid Credentials")

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        if register_user(username, email, password):
            flash("Registration Successful")
            return redirect(url_for("auth.login"))

        flash("Email already exists")

    return render_template("register.html")


@auth_bp.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("auth.login"))