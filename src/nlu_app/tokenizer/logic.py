from typing import List
import string

# Define punctuation as a constant at the module level for efficiency.
# string.punctuation provides a standard set of punctuation characters.
PUNCTUATION_TO_REMOVE = string.punctuation

def tokenize_text(text: str) -> List[str]:
    """
    Performs basic tokenization on a given text.

    The process includes:
    1. Lowercasing the text.
    2. Removing all standard punctuation.
    3. Splitting the text into words based on whitespace.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        List[str]: A list of cleaned tokens (words).
    """
    if not text:
        return []

    # Lowercase the text
    processed_text = text.lower()

    # Remove punctuation
    # Create a translation table to remove all punctuation characters
    translator = str.maketrans('', '', PUNCTUATION_TO_REMOVE)
    processed_text = processed_text.translate(translator)

    # Split into words (tokens) using whitespace as a delimiter
    # .split() also handles multiple spaces and leading/trailing whitespace gracefully.
    tokens = processed_text.split()

    return tokens
