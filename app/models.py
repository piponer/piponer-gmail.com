from app import db
from datetime import datetime
from flask_login import UserMixin

class Users(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    birthday = db.Column(db.String(100), nullable=False)
    career = db.Column(db.String(100), nullable=False)

class UserWords(db.Model):
    __tablename__ = 'userwords'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content = db.Column(db.Text, nullable=False)
    time = db.Column(db.Text)
    answer = db.Column(db.Text)
    answer_time = db.Column(db.Text)
    talker_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    talker = db.relationship('Users', backref=db.backref('userwords'))

class Feedback(db.Model):
    __tablename__ = 'feedback'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    category = db.Column(db.String(50))
    feedback = db.Column(db.String(1200))
    rating = db.Column(db.Integer)

class ContactUs(db.Model):
    __tablename__ = 'contactus'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    message = db.Column(db.String(1200))

class News(db.Model):
    __tablename__ = 'news'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    news = db.Column(db.Text)

class Ratings(db.Model):
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    rating = db.Column(db.Integer)

class Questions(db.Model):
    __tablename__ = 'questions'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question = db.Column(db.Integer)
    talker_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    talker = db.relationship('Users', backref=db.backref('questions'))

db.create_all()