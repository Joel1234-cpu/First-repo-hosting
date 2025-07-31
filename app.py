from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import pickle, os

load_dotenv()

# 🌐 ENV: 'local' for development, 'production' for Railway
ENV = os.getenv("ENV", "local")

app = Flask(__name__)

# 🧠 Choose the database based on environment
if ENV == "production":
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("RAILWAY_DB_URL")
    print("🔗 Connected to Railway Database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("LOCAL_DB_URL")
    print("🖥️ Connected to Local Database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 📦 SQLAlchemy model
class Predictions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    predicted_gender = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# 🤖 Load gender prediction model
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

        # Simple hardcoded predictions
        common_names = {'john': 'male', 'mary': 'female'}
        if name in common_names:
            prediction = common_names[name]
            confidence = 95.0
        else:
            vector = vectorizer.transform([name])
            proba = model.predict_proba(vector)[0]
            prediction = model.predict(vector)[0]
            confidence = round(float(max(proba)) * 100, 2)

        # ✅ Store prediction
        entry = Predictions(name=name, predicted_gender=prediction, confidence=confidence)
        db.session.add(entry)
        db.session.commit()

        return render_template('index.html', name=name, gender=prediction, confidence=confidence)
    except Exception as e:
        return render_template('index.html', error=f"❌ Error: {str(e)}")

# 🚀 Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates tables if not exist
    app.run(debug=True)
