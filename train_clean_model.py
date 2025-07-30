import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

print("🔄 Loading cleaned dataset...")
# Load the dataset
data = pd.read_csv('names_dataset.csv')

# Clean column names by stripping whitespace
data.columns = [col.strip() for col in data.columns]

# Print column names to verify
print("Available columns:", data.columns.tolist())

# Clean the data
data['Name'] = data['Name'].str.strip()
data['Gender'] = data['Gender'].str.strip()

# Normalize gender values
data['Gender'] = data['Gender'].str.lower()
data['Gender'] = data['Gender'].map({
    'm': 'male', 'male': 'male', '1': 'male', 'boy': 'male',
    'f': 'female', 'female': 'female', '0': 'female', 'girl': 'female'
})

# Remove any rows with missing values
data = data.dropna()

# Remove any remaining invalid entries
data = data[data['Gender'].isin(['male', 'female'])]

print(f"✅ Dataset size: {len(data)}")
print(f"✅ Gender distribution:\n{data['Gender'].value_counts()}")

# Vectorization with character n-grams
print("🔄 Creating character features...")
vectorizer = CountVectorizer(
    analyzer='char_wb',  # Character-based n-grams with word boundaries
    ngram_range=(2, 4),  # Use 2-4 character sequences
    lowercase=True
)

X = vectorizer.fit_transform(data['Name'])
y = data['Gender']

print(f"✅ Feature matrix shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Model training
print("🔄 Training model...")
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Test with some common names
test_names = ['srujeeth', 'divyanshu', 'michael', 'aditya', 'sabir', 'joel', 
              'roshni', 'deena', 'joice', 'david', 'emma', 'james', 'olivia']
print(f"\n🧪 Testing with common names:")
for name in test_names:
    vector = vectorizer.transform([name])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = max(proba) * 100
    print(f"{name.title()}: {prediction} ({confidence:.1f}%)")

# Save model and vectorizer
print("\n💾 Saving model and vectorizer...")
pickle.dump(model, open('gender_model_clean.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer_clean.pkl', 'wb'))

print("✅ Model and vectorizer saved successfully!")
print("📁 Files created: gender_model_clean.pkl, vectorizer_clean.pkl")