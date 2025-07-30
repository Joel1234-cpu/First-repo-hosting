import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

# Load and clean dataset
data = pd.read_csv('names_dataset.csv')
data.columns = [col.strip().lower() for col in data.columns]

# Normalize values
data['gender'] = data['gender'].astype(str).str.strip().str.lower()
data['gender'] = data['gender'].map({
    'm': 'male', 'male': 'male', '1': 'male', 'boy': 'male',
    'f': 'female', 'female': 'female', '0': 'female', 'girl': 'female'
})
data.dropna(subset=['Name', 'Gender'], inplace=True)

print(f"✅ Cleaned dataset size: {len(data)}")
print("✅ Gender values:", data['gender'].unique())

# Vectorization
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(data['name'].str.lower())
y = data['gender']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('gender_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("\n✅ Model and vectorizer saved successfully!")
