from flask import Flask, request, render_template
import pickle
import mysql.connector
import os  # Import the os module

app = Flask(__name__)

# Load model and vectorizer
try:
    model = pickle.load(open('gender_model_clean.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer_clean.pkl', 'rb'))
    print("✅ Loaded cleaned model")
except:
    model = pickle.load(open('gender_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("⚠️ Loaded original model (cleaned model not found)")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name'].lower().strip()
    
    # Basic validation
    if not name or len(name) < 2:
        return render_template('index.html', error="Please enter a valid name (at least 2 characters)")
    
    # Common name corrections for better accuracy
    common_names = {
        'john': 'male', 'michael': 'male', 'david': 'male', 'james': 'male',
        'mary': 'female', 'sarah': 'female', 'emma': 'female', 'olivia': 'female',
        'william': 'male', 'elizabeth': 'female', 'robert': 'male', 'jennifer': 'female'
    }
    
    # Check if it's a common name first
    if name in common_names:
        prediction = common_names[name]
        confidence = 95.0
    else:
        # Use model prediction
        vector = vectorizer.transform([name])
        proba = model.predict_proba(vector)[0]
        prediction = model.predict(vector)[0]
        confidence = round(max(proba) * 100, 2)

    try:
        # Use environment variables for database credentials
        db_host = os.environ.get('DB_HOST', 'localhost')  # Default to localhost if not set
        db_user = os.environ.get('DB_USER', 'root')       # Default to root if not set
        db_password = os.environ.get('DB_PASSWORD')       # Ensure this is set in your environment
        db_name = os.environ.get('DB_NAME', 'Gender_Prediction') # Default to Gender_Prediction if not set

        if not db_password:
            raise ValueError("Database password not found in environment variables.")

        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        # Ensure table exists
        try:
            cursor.execute("SELECT 1 FROM predictions LIMIT 1")
            cursor.fetchone()
        except mysql.connector.Error as err:
            if err.errno == 1146:
                cursor.execute("""
                    CREATE TABLE predictions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        predicted_gender VARCHAR(10) NOT NULL,
                        confidence FLOAT
                    )
                """)
                conn.commit()

        # Insert prediction with confidence
        sql = "INSERT INTO predictions (name, predicted_gender, confidence) VALUES (%s, %s, %s)"
        cursor.execute(sql, (name, prediction, confidence))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        return f"Connection or query error: {e}"

    return render_template('index.html', name=name, gender=prediction, confidence=confidence)




if __name__ == '__main__':
    app.run(debug=True)
