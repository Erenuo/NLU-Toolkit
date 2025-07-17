import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

print("--- Prediction Script Initialized ---")

# --- 1. Load Model and Vectorizer ---
try:
    model = joblib.load('saved_model/sentiment_model.joblib')
    vectorizer = joblib.load('saved_model/tfidf_vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("\nError: Model or vectorizer files not found.")
    print("Please run 'Review_Analyzer_Train.py' first to train and save the model.")
    exit()

# --- 2. Setup NLTK and Preprocessing Function ---
# This setup must be IDENTICAL to the one used in the training script.
nltk_data_path = './nltk_data'
if not os.path.exists(nltk_data_path):
    # If the folder doesn't exist, the resources can't be loaded.
    print(f"NLTK data directory not found at '{nltk_data_path}'.")
    print("Please run the training script first to download NLTK resources.")
    exit()
nltk.data.path.append(nltk_data_path)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and prepares text for modeling."""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# --- 3. Prediction Function ---
def predict_sentiment(review_text):
    """
    Takes a raw review string and predicts its sentiment.
    """
    if not review_text.strip():
        return "Cannot predict sentiment for an empty review."
        
    # Preprocess the input text
    processed_text = preprocess_text(review_text)
    
    # Vectorize the text using the loaded TF-IDF vectorizer
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict using the loaded model
    prediction = model.predict(vectorized_text)
    
    # Return the result
    return prediction[0].capitalize()

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("\n--- IMDb Review Sentiment Analyzer ---")
    print("Enter a movie review to analyze its sentiment.")
    print("Type 'exit' to quit the program.")

    while True:
        # Get user input
        user_review = input("\nEnter your review: ")
        
        # Check for exit condition
        if user_review.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
        
        # Get the prediction
        sentiment = predict_sentiment(user_review)
        
        # Display the result
        print(f"-> Predicted Sentiment: {sentiment}")

