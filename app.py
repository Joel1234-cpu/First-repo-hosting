from flask import Flask, request, render_template
import pickle
import mysql.connector
import os

app = Flask(__name__)

# 🔄 Load model and vectorizer
try:
    model = pickle.load(open('gender_model_clean.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer_clean.pkl', 'rb'))
    print("✅ Loaded cleaned model")
except:
    model = pickle.load(open('gender_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("⚠️ Loaded fallback model (cleaned model not found)")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name'].lower().strip()
        print(f"✅ Received name: {name}")

        if not name or len(name) < 2:
            return render_template('index.html', error="Please enter a valid name (at least 2 characters)")

        # Common name shortcut
        common_names = {
            'john': 'male', 'michael': 'male', 'david': 'male', 'james': 'male',
            'mary': 'female', 'sarah': 'female', 'emma': 'female', 'olivia': 'female',
            'william': 'male', 'elizabeth': 'female', 'robert': 'male', 'jennifer': 'female'
        }

        if name in common_names:
            prediction = common_names[name]
            confidence = 95.0
        else:
            vector = vectorizer.transform([name])
            proba = model.predict_proba(vector)[0]
            prediction = model.predict(vector)[0]
            confidence = round(float(max(proba)) * 100, 2)

        # 🌐 Connect to MySQL using Railway ENV variables
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = int(os.environ.get('DB_PORT', 3306))
        db_user = os.environ.get('DB_USER', 'root')
        db_password = os.environ.get('DB_PASSWORD')
        db_name = os.environ.get('DB_NAME', 'Gender_Prediction')

        if not db_password:
            raise ValueError("Database password not found in environment variables.")

        print("🔌 Connecting to DB...")
        conn = mysql.connector.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        # ✅ Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                predicted_gender VARCHAR(10) NOT NULL,
                confidence FLOAT
            )
        """)
        conn.commit()

        # 📥 Insert prediction
        sql = "INSERT INTO predictions (name, predicted_gender, confidence) VALUES (%s, %s, %s)"
        cursor.execute(sql, (name, prediction, confidence))
        conn.commit()

        cursor.close()
        conn.close()
        print("📦 Prediction saved to DB.")

        return render_template('index.html', name=name, gender=prediction, confidence=confidence)

    except Exception as e:
        print("❌ Error during /predict:", e)
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
