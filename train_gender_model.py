import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


# Load dataset
data = pd.read_csv('names_dataset.csv')

data['Name'] = data['Name'].str.lower()

# Extract last 1, 2, and 3 letters
data['last_1'] = data['Name'].str[-1:]
data['last_2'] = data['Name'].str[-2:]
data['last_3'] = data['Name'].str[-3:]

# Encode characters to numeric features
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()


data['last_1_enc'] = le1.fit_transform(data['last_1'])
data['last_2_enc'] = le2.fit_transform(data['last_2'])
data['last_3_enc'] = le3.fit_transform(data['last_3'])


X = data[['last_1_enc', 'last_2_enc', 'last_3_enc']]
y = data['Gender']

# Encode target labels
gender_le = LabelEncoder()
y = gender_le.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
pickle.dump(model, open('gender_suffix_model.pkl', 'wb'))
pickle.dump((le1, le2, le3, gender_le), open('suffix_encoders.pkl', 'wb'))


print("Model trained on suffixes and saved successfully!")
