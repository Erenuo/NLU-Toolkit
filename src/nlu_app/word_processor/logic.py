from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from typing import Dict, List
import nltk

# --- Module-Level Variables ---
# These will be initialized once by the load function.
porter_stemmer = None
wordnet_lemmatizer = None

def load_word_processing_tools():
    """
    Downloads necessary NLTK resources and initializes the stemmer and lemmatizer.
    This function should be called once when the application starts.
    """
    global porter_stemmer, wordnet_lemmatizer
    
    print("--- Loading Word Processing Tools (Stemmer/Lemmatizer) ---")
    try:
        # Download necessary NLTK data. Using quiet=True to avoid verbose output.
        print("Checking NLTK resources (punkt, wordnet, omw-1.4, averaged_perceptron_tagger)...")
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK resources are up to date.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        raise e

    # Initialize the tools and assign them to the global variables
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    print("Porter Stemmer and WordNet Lemmatizer initialized successfully.")

def _get_wordnet_pos(word: str) -> str:
    """
    Internal helper function to map NLTK's POS tag to a format WordNetLemmatizer understands.
    """
    # Get the part-of-speech tag for the word
    tag = nltk.pos_tag([word])[0][1][0].upper()
    # Create a mapping dictionary
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    # Return the mapped tag, defaulting to NOUN if not found
    return tag_dict.get(tag, wordnet.NOUN)

def process_word(word: str) -> Dict[str, str]:
    """
    Performs stemming and lemmatization on a single word.

    Args:
        word (str): The word to process.

    Returns:
        Dict[str, str]: A dictionary containing the original, stemmed, and lemmatized forms.
    """
    if porter_stemmer is None or wordnet_lemmatizer is None:
        raise RuntimeError("Word processing tools are not loaded. Please run 'load_word_processing_tools' at startup.")

    # Clean and standardize the input word
    word = word.strip().lower()
    if not word:
        return {"original": "", "stemmed": "", "lemmatized": ""}

    # Perform stemming
    stemmed_word = porter_stemmer.stem(word)
    
    # Perform lemmatization using the POS tag for better accuracy
    lemmatized_word = wordnet_lemmatizer.lemmatize(word, _get_wordnet_pos(word))
    
    return {
        "original": word,
        "stemmed": stemmed_word,
        "lemmatized": lemmatized_word
    }

def process_word_list(words: List[str]) -> List[Dict[str, str]]:
    """
    Performs stemming and lemmatization on a list of words.

    Args:
        words (List[str]): The list of words to process.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing results for a word.
    """
    # Process each word in the list using the single-word processing function
    return [process_word(word) for word in words if word.strip()]
