from datetime import datetime
from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_login import login_required, current_user, LoginManager, login_user, logout_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, func

import dialogflow
import os

from app import login_manager, app, db
from app.models import Users, UserWords, Feedback, ContactUs, News, Ratings, Questions
from app.dumb_inference import core_inference

@login_manager.user_loader 
def load_user(user_id):
    return Users.query.get(int(user_id))

@login_manager.unauthorized_handler
def handle_needs_login():
    news = News.query.all()
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    news = News.query.all()
    return redirect(url_for('home'))

@app.route('/', methods=['POST', 'GET'])
def login():
    news = News.query.all()
    total, avg = get_ratings_statistics()
    num_users = db.session.query(func.count(Users.id)).scalar()
    total_user_questions, total_questions = get_questions_statistics()

    if request.method == 'GET':
        # redirect user to their chat page if they are already logged in
        if (current_user.is_authenticated):
            username = current_user.username
            return redirect(url_for('chat', userID=username))

        return render_template('home.html', news_list=reversed(news), avg_rating=avg, num_rating=total, num_users=num_users, total_questions=int(total_questions))
    else:
        emailaddress = request.form.get('emailaddress')
        password = request.form.get('password')
        remember = request.form.get('remember-me')
        user = Users.query.filter(Users.username == emailaddress, Users.password == password).first()

        # redirect user to chat page if valid user
        if user:
            login_user(user, remember=remember)
            return redirect(url_for('chat', userID=emailaddress))
        
        else:
            message="Invalid email address or password!"
            return render_template('home.html', message=message, news_list=reversed(news), avg_rating=avg, num_rating=total, num_users=num_users, total_questions=int(total_questions))

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        emailaddress = request.form.get('emailaddress')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        birthday = request.form.get('birthday')
        gender = request.form.get('gender')
        career = request.form.get('career')

        if emailaddress != "" and password1 != "" and password2 != "" and birthday != "" and gender != "" and career != "":
            user = Users.query.filter(Users.username == emailaddress).first()
            if user:
                message="This email is not available!"
                return render_template('register.html', message=message)
            else:
                if password1 != password2:
                    message1="The passwords do not match!"
                    return render_template('register.html', message=message1)
                else:
                    user = Users(username=emailaddress, password=password1, gender=gender, birthday=birthday,
                                 career=career)
                    db.session.add(user)
                    db.session.commit()
                    return redirect(url_for('home'))
        else:
            message = "Please fill in all fields!"
            return render_template('register.html', message=message)

@app.route('/')
def home():
    # redirect user to their chat page if they are already logged in
    if (current_user.is_authenticated):
        username = current_user.username
        return redirect(url_for('chat', userID=username))

    # otherwise redirect to home page
    news = News.query.all()
    total, avg = get_ratings_statistics()
    num_users = db.session.query(func.count(Users.id)).scalar()
    total_user_questions, total_questions = get_questions_statistics()
    return render_template('home.html', news_list=reversed(news), avg_rating=avg, num_rating=total, num_users=num_users, total_questions=int(total_questions))

@app.route('/chat/<userID>', methods=['GET'])
@login_required
def chat(userID):
    """
        Render current user's chat page
    """
    userID = userID
    # user tried to access another user's chat page 
    if (userID != current_user.username):
        return redirect(url_for('chat', userID=current_user.username))

    user_id = current_user.get_id()
    context = dict()
    user = Users.query.filter(Users.id == user_id).first()
    total_user_questions, total_questions = get_questions_statistics()
    context = {
        'questions': UserWords.query.filter(user_id == UserWords.talker_id)
    }
    
    return render_template('chat.html', **context,userID=userID, total_user_questions=int(total_user_questions))

@app.route('/feedback', methods=['POST', 'GET'])
def receive_feedback():
    """
        Add user input from Feedback page to database
    """
    if request.method == 'POST':
        data = request.get_json(silent=False)

        category = data['selected_category']
        feedback = data['user_feedback']
        rating = data['user_rating']

        if (feedback != '' or rating != 0):
            feedback_ = Feedback(category=category, feedback=feedback, rating=rating)
            db.session.add(feedback_)

            # keep track of ratings from all users
            if (rating != 0):
                rating_ = Ratings(rating=rating)
                db.session.add(rating_)

            db.session.commit()
        return 'OK', 200
        
    return redirect('/')

@app.route('/contact-us', methods=['POST', 'GET'])
def contact_us():
    """
        Add user input from ContactUs page to the database
    """
    if request.method == 'POST':
        data = request.get_json(silent=False)
        
        name = data['contact_name']
        email = data['contact_email']
        message = data['contact_message']

        if name != '' and email != '' and message != '':
            contact_us = ContactUs(name=name, email=email, message=message)
            db.session.add(contact_us)
            db.session.commit()
        return 'OK', 200
        
    return redirect('/')

@app.route('/query', methods=['POST', 'GET'])
@login_required
def query():
    """
        Perform backend algorithm to get an answer
    """
    if request.method == 'POST':
        input = request.get_json(silent=False)
        sentence = input['sentence']
        out_dict = core_inference(sentence)    # run algorithm to get answer
        #out_dict = {'a', 'kl', 'label'}
        if out_dict['label'] and ('comp' in sentence.lower() or 'unsw' in sentence.lower()):
            return {'answer': "I don't know."}

        return {'answer': out_dict['a']}

def detect_intent_texts(project_id, session_id, text, language_code):
    """
        Dialogflow helper function that sends and receives answer
    """
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)
    if text:
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        return response.query_result.fulfillment_text

@app.route('/send_message', methods=['GET', 'POST'])
@login_required
def send_message():
    """
        Queries Dialogflow for an answer to a question
    """
    data = request.get_json(silent=False)
    message = data['message']
    project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    fulfillment_text = detect_intent_texts(project_id, "unique", message, 'en')
    response_text = { "message":  fulfillment_text }

    return jsonify(response_text)

@app.route('/add-sentence-to-db', methods=['POST'])
@login_required
def add_question_to_db():
    """
        Adds question, answer pairs to database
    """
    data = request.get_json(silent=False)
    user_id = current_user.get_id()
    user = Users.query.filter(Users.id == user_id).first()
    content = data['sentence']
    answer = data['answer']
    question_date = data['question_date']
    answer_date = data['answer_date']

    if content != "":
        # add question, answer pair
        words = UserWords(content=content, answer=answer, time=question_date, answer_time=answer_date)
        words.talker_id = user.id
        db.session.add(words)

        # increment question count for all users 
        count = db.session.query(Questions).filter_by(talker_id=user_id).first()
        if count is None:
            db.session.add(Questions(question=1, talker_id=user_id))
        else:
            count.question = int(count.question) + 1
        db.session.commit()
    return 'OK', 200

@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    """
        Clears all questions and answers from database
    """
    user_id = current_user.get_id()
    UserWords.query.filter(user_id == UserWords.talker_id).delete()
    db.session.commit()
    return 'OK', 200

def get_ratings_statistics():
    avg = db.session.query(func.avg(Ratings.rating)).scalar()
    total = db.session.query(func.count(Ratings.rating)).scalar()
    if avg is None:
        avg = 0
    if total is None:
        total = 0
    return total, avg

def get_questions_statistics():
    user_id = current_user.get_id()
    total_user_questions = db.session.query(func.sum(Questions.question)).filter(Questions.talker_id==user_id).scalar()
    total_questions = db.session.query(func.sum(Questions.question)).scalar()  
    if total_questions is None:
        total_questions = 0
    if total_user_questions is None:
        total_user_questions = 0 
    return total_user_questions, total_questions

if __name__ == '__main__':
    app.run(debug=True)
