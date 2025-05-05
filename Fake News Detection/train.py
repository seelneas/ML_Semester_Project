# Import necessary libraries
import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load and label data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')
df_fake['class'] = 0
df_true['class'] = 1

df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

df = pd.concat([df_fake, df_true], axis=0)
df = df.drop(['title', 'subject', 'date'], axis=1)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'$.*?$', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|[www.\S+](http://www.\S+)', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df["text"] = df["text"].apply(wordopt)

# Features & labels
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Vectorization
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

# Models
models = {
    "SVM": SVC(kernel='linear'),
    "DT": DecisionTreeClassifier(),
    "GBC": GradientBoostingClassifier(random_state=0),
    "RFC": RandomForestClassifier()
}

# Train, evaluate, save
for name, model in models.items():
    model.fit(xv_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")
    
    # Predict
    y_pred = model.predict(xv_test)

    # Confusion Matrix and Classification Report
    print(f"\n📊 {name} Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

print("✅ Models and vectorizer saved successfully.")
