# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:23:39 2019

@author: 63184
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_required, current_user, LoginManager, login_user, logout_user, UserMixin
from sqlalchemy import create_engine

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "No-Idea-fed233c27de5.json"
os.environ["DIALOGFLOW_PROJECT_ID"] = "no-idea-imqykq"

app = Flask(__name__)

DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = 'put your MySQL username here'
PASSWORD = 'put your MySQL password here'
HOST = 'localhost'
PORT = '3306'
DATABASE = 'test'

app.config.from_mapping (
    SECRET_KEY= b'\xd5\x14\xffrm<\xe9|\x8d\xc9\xd8\xbf\x1cq\xe0\xaeZc\xb1\x9dSj\xe2+',
    SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}?charset=UTF8MB4".format(DIALECT, DRIVER, USERNAME, PASSWORD, HOST, PORT, DATABASE),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
)

# create database
mysql_engine = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}'.format(USERNAME, PASSWORD, HOST, PORT))
existing_databases = mysql_engine.execute("SHOW DATABASES;")
existing_databases = [d[0] for d in existing_databases]
if DATABASE not in existing_databases:
    mysql_engine.execute("CREATE DATABASE {0}".format(DATABASE))
    print("Created database {0}".format(DATABASE))
db_engine = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(USERNAME, PASSWORD, HOST, PORT, DATABASE))
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

from app import route