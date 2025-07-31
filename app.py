from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import pickle, os

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DB_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    predicted_gender = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# Load model
try:
    model = pickle.load(open('gender_model_clean.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer_clean.pkl', 'rb'))
    print("✅ Cleaned model loaded")
except:
    model = pickle.load(open('gender_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("⚠️ Fallback model loaded")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name'].lower().strip()
        if not name or len(name) < 2:
            return render_template('index.html', error="Name too short!")

        common_names = {'john': 'male', 'mary': 'female'}
        if name in common_names:
            prediction = common_names[name]
            confidence = 95.0
        else:
            vector = vectorizer.transform([name])
            proba = model.predict_proba(vector)[0]
            prediction = model.predict(vector)[0]
            confidence = round(float(max(proba)) * 100, 2)

        # ✅ Insert using SQLAlchemy
        entry = Prediction(name=name, predicted_gender=prediction, confidence=confidence)
        db.session.add(entry)
        db.session.commit()

        return render_template('index.html', name=name, gender=prediction, confidence=confidence)
    except Exception as e:
        return render_template('index.html', error=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()    # ✅ Ensure tables are created locally
    app.run(debug=True)
