from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from typing import List
import nltk
import re


# --- Module-Level Variables ---
# These will be initialized by the load function.
stop_words = None
lemmatizer = None

def load_summarizer_tools():
    """
    Downloads necessary NLTK resources and initializes tools for summarization.
    This should be called once at application startup.
    """
    global stop_words, lemmatizer
    
    print("--- Loading Extractive Summarizer Tools ---")
    try:
        # Download necessary NLTK data quietly.
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("NLTK resources for summarizer are up to date.")
    except Exception as e:
        print(f"Error downloading NLTK data for summarizer: {e}")
        raise e

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("Summarizer tools (stopwords, lemmatizer) initialized.")

def _preprocess_text(text: str) -> List[str]:
    """
    Internal helper function to clean, tokenize, and lemmatize text.
    Returns a list of processed words.
    """
    if stop_words is None or lemmatizer is None:
        raise RuntimeError("Summarizer tools are not loaded. Please run 'load_summarizer_tools' at startup.")
        
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

def generate_extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Generates an extractive summary of the given text based on word frequency.

    Args:
        text (str): The input text to be summarized.
        num_sentences (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary.
    """
    # Tokenize the original text into sentences
    original_sentences = sent_tokenize(text)

    # If the text is already short enough, return it as is.
    if len(original_sentences) <= num_sentences:
        return text

    # Calculate word frequencies for the entire document
    all_processed_words = _preprocess_text(text)
    word_frequencies = Counter(all_processed_words)

    # Score each sentence
    sentence_scores = {}
    for i, sentence in enumerate(original_sentences):
        processed_sentence_words = _preprocess_text(sentence)
        if not processed_sentence_words:
            continue  # Skip empty or stopword-only sentences

        # Calculate score based on the sum of frequencies of its words
        score = sum(word_frequencies.get(word, 0) for word in processed_sentence_words)
        
        # Normalize score by sentence length to avoid bias towards longer sentences
        sentence_scores[i] = score / len(processed_sentence_words)

    # Select the top N sentences
    # Get the original indices of the sentences with the highest scores
    sorted_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    top_sentence_indices = sorted(sorted_sentence_indices[:num_sentences])

    # Build the summary by joining the top sentences in their original order
    summary = ' '.join(original_sentences[i] for i in top_sentence_indices)
    return summary
