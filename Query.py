from app import login_manager, app, db
from app.models import Users, UserWords, Feedback, ContactUs, News, Ratings, Questions
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import csv

def save_contact_us():
    """
        Export contactUs table to csv
    """
    contactUs = ContactUs.query.all()
    with open('contactUs.csv','w') as csv_in: 
        writer = csv.writer(csv_in)
        for contact in contactUs:
            # print(contact.name, contact.email, contact.message)
            if contact.email == '':
                continue
            writer.writerow([contact.name, contact.email, contact.message])
    csv_in.close()

def save_feedback():
    """
        Export feedback table to csv
    """
    feedbacks = Feedback.query.all()
    with open('feedback.csv','w') as csv_in: 
        writer = csv.writer(csv_in)
        for feedback in feedbacks:
            # print(feedback.category, feedback.feedback, feedback.rating)
            f = feedback.feedback if feedback.feedback != '' else 'No feedback'
            r = feedback.rating if feedback.rating != 0 else 'No Rating'
            writer.writerow([feedback.category, f, r])
    csv_in.close()

def save_news():
    """
        Export news table to csv
    """
    news_list = News.query.all()
    with open('news.csv','w') as csv_in: 
        writer = csv.writer(csv_in)
        for news in news_list:
            # print(news.news)
            writer.writerow([news.news])
    csv_in.close()

def delete_contact_records():
    db.session.query(ContactUs).delete()
    db.session.commit()

def delete_feedback_records():
    db.session.query(Feedback).delete()
    db.session.commit()

def delete_news_records():
    db.session.query(News).delete()
    db.session.commit()

def insert_news(news_list):
    for news in news_list:
        db.session.add(News(news=news))
        db.session.commit()

def insert_news_from_csv(csv_name):
    news_list = []
    with open(csv_name, 'r') as csv_in:
        reader = csv.reader(csv_in)
        lines = list(reader)
        for line in lines:
            if len(line) == 0:
                continue
            db.session.add(News(news=line[0]))
    db.session.commit()
    csv_in.close()

def insert_news_from_txt(file):
    with open(file, 'r') as f:
        for line in f:
            # print(line)
            if line == '':
                continue
            db.session.add(News(news=line))
    db.session.commit()
    f.close()

if __name__ == '__main__':
    query = input("Access ContactUs (A), Feedback (B) or News (C)? ")
    if query == 'A':
        action = input("Export records (A) or delete records (B)? ")
        if action == 'A':
            save_contact_us()
        elif action == 'B':
            delete_contact_records()
        else:
            print("Please enter A or B")

    elif query == 'B':
        action = input("Export records (A) or delete records (B)? ")
        if action == 'A':
            save_feedback()
        elif action == 'B':
            delete_feedback_records()
        else: 
            print("Please enter A or B")

    elif query == 'C':
        action = input("Insert news (A), export records (B) or delete records (C)? ")
        if action == 'A':
            method = input("Insert from file (enter name) or manually enter (m)? ")
            if method == 'm':
                news_list = []
                news = ''
                while True:
                    news = input("Enter news (e to exit): ")
                    if (news == 'e'):
                        break
                    news_list.append(news)
                insert_news(news_list)            
                save_news()
            elif '.csv' in method:
                insert_news_from_csv(method)
            else: 
                insert_news_from_txt(method)
        elif action == 'B':
            save_news()
        elif action == 'C':
            delete_news_records()
        else:
            print("Please enter A, B or C")
    
    else:
        print("Please enter A, B or C")
    

