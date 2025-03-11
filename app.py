from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your_secret_key'
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    results = db.relationship('Result', backref='user', lazy=True)

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    questions = db.relationship('Question', backref='quiz', lazy=True)

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    options = db.Column(db.JSON, nullable=False)
    correct_answer = db.Column(db.String(100), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)

# Recommendation System
class QuizRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def train(self, questions):
        texts = [q.text + " " + q.category for q in questions]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def recommend(self, user_history, all_questions, top_n=5):
        history_texts = [q.text + " " + q.category for q in user_history]
        history_vec = self.vectorizer.transform(history_texts)
        sim_scores = cosine_similarity(history_vec, self.tfidf_matrix)
        avg_scores = np.mean(sim_scores, axis=0)
        top_indices = np.argsort(avg_scores)[-top_n:][::-1]
        return [all_questions[i] for i in top_indices]

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

class RegisterForm(LoginForm):
    submit = SubmitField('Register')

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    user_history = Result.query.filter_by(user_id=1).all()
    all_questions = Question.query.all()
    recommender = QuizRecommender()
    recommender.train(all_questions)
    recommended = recommender.recommend(user_history, all_questions)
    return render_template('dashboard.html', quizzes=Quiz.query.all(), recommended=recommended)

@app.route('/take_quiz/<int:quiz_id>', methods=['GET', 'POST'])
def take_quiz(quiz_id):
    quiz = Quiz.query.get_or_404(quiz_id)
    if request.method == 'POST':
        score = calculate_score(quiz, request.form)
        return redirect(url_for('dashboard'))
    return render_template('take_quiz.html', quiz=quiz)

def calculate_score(quiz, answers):
    correct = 0
    for question in quiz.questions:
        if answers.get(str(question.id)) == question.correct_answer:
            correct += 1
    return (correct / len(quiz.questions)) * 100

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)