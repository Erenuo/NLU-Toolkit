import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
import os
import joblib # Used for saving the model and vectorizer

print("--- Training Script Initialized ---")

# --- 1. NLTK Setup ---
# Set NLTK data path to a writable directory
nltk_data_path = './nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Function to download NLTK data if not found
def download_nltk_resource(resource, resource_name):
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        print(f"Downloading NLTK resource '{resource_name}'...")
        nltk.download(resource_name, download_dir=nltk_data_path)

download_nltk_resource('stopwords', 'stopwords')
download_nltk_resource('wordnet', 'wordnet')

# --- 2. Load and Preprocess Data ---
# Load the IMDB dataset
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found.")
    print("Please download the dataset and place it in the same directory.")
    exit()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and prepares text for modeling."""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text) # Remove HTML line breaks
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Preprocessing text data...")
df['processed_text'] = df['review'].apply(preprocess_text)
print("Preprocessing complete.")

# --- 3. Feature Engineering and Data Splitting ---
X_text = df['processed_text']
y = df['sentiment']

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and fit the vectorizer ON THE TRAINING DATA ONLY
vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.7)
print("Fitting vectorizer on training data...")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("Vectorizer fitting and data transformation complete.")

# --- 4. Model Training ---
model = MultinomialNB(alpha=0.1)
print("Training the Multinomial Naive Bayes model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation on Test Set ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Model and Vectorizer ---
# Create a directory to store the model files if it doesn't exist
os.makedirs('saved_model', exist_ok=True)

# Save the trained model
model_filename = 'saved_model/sentiment_model.joblib'
joblib.dump(model, model_filename)
print(f"\nModel saved to {model_filename}")

# Save the fitted vectorizer
vectorizer_filename = 'saved_model/tfidf_vectorizer.joblib'
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer saved to {vectorizer_filename}")
print("\n--- Training Script Finished ---")
