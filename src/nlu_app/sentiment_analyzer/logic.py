from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib
import nltk
import re
import os

# --- Module-Level Variables ---
# These will be loaded once by the load_sentiment_model function
model = None
vectorizer = None
lemmatizer = None
stop_words = None

def _preprocess_text(text: str) -> str:
    """
    Cleans and prepares text for modeling.
    This is an internal helper function.
    """
    global lemmatizer, stop_words
    
    # Check if resources are loaded
    if lemmatizer is None or stop_words is None:
        raise RuntimeError("NLTK resources are not initialized. Please run 'load_sentiment_model' first.")

    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def load_sentiment_model(model_dir: str = 'saved_model', nltk_data_dir: str = './nltk_data'):
    """
    Loads the sentiment model, vectorizer, and NLTK resources into memory.
    This function should be called once when the application starts.
    """
    global model, vectorizer, lemmatizer, stop_words
    
    print("--- Loading Sentiment Analysis Model and Resources ---")
    
    # Load Model and Vectorizer
    try:
        model_path = os.path.join(model_dir, 'sentiment_model.joblib')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure models exist in the 'saved_model' directory.")
        # Re-raise the exception to stop the application from starting incorrectly
        raise e

    # Setup NLTK
    if not os.path.exists(nltk_data_dir):
        error_msg = f"NLTK data directory not found at '{nltk_data_dir}'."
        print(error_msg)
        raise FileNotFoundError(error_msg)
        
    nltk.data.path.append(nltk_data_dir)
    
    # Initialize lemmatizer and stopwords after setting the path
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    print("NLTK resources initialized.")

def predict_sentiment(review_text: str) -> str:
    """
    Takes a raw review string and predicts its sentiment (Positive/Negative).
    """
    global model, vectorizer

    # Ensure models are loaded before predicting
    if model is None or vectorizer is None:
        raise RuntimeError("Model is not loaded. Please run 'load_sentiment_model' at application startup.")

    if not review_text.strip():
        return "Cannot predict sentiment for an empty review."
        
    # Preprocess and predict
    processed_text = _preprocess_text(review_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    
    # Return the result in a consistent format
    return str(prediction[0]).capitalize()
