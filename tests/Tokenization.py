# This script demonstrates the basic steps of text tokenization.
# Tokenization is the process of breaking down a text into smaller units called tokens (usually words or subwords).

def simple_tokenize(text):
    """
    Performs a step-by-step tokenization process on the given text.
    """
    print(f"Original Text: \"{text}\"\n")

    # Step 1: Lowercasing
    # Convert all characters in the text to lowercase. This helps to treat
    # words like "The" and "the" as the same token.
    print("Step 1: Lowercasing")
    lowercased_text = text.lower()
    print(f"Lowercased Text: \"{lowercased_text}\"\n")

    # Step 2: Punctuation Removal (Simple)
    # Remove common punctuation marks. This prevents punctuation from being
    # part of the tokens (e.g., "word." vs "word").
    # We'll use a simple approach here, iterating through common punctuation.
    print("Step 2: Punctuation Removal")
    punctuation_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    text_without_punctuation = lowercased_text
    for char in punctuation_to_remove:
        text_without_punctuation = text_without_punctuation.replace(char, "")
    print(f"Text without Punctuation: \"{text_without_punctuation}\"\n")

    # Step 3: Splitting into words (Whitespace Tokenization)
    # Split the cleaned text into a list of words using whitespace as a delimiter.
    # This is a very common and basic form of tokenization.
    print("Step 3: Splitting into Words (Whitespace Tokenization)")
    tokens = text_without_punctuation.split()
    print(f"Tokens: {tokens}\n")

    print("Tokenization Process Complete!")
    return tokens

# --- Example Usage ---
sample_text_1 = "Hello, World! This is a sample sentence for tokenization."
print("--- Tokenization Example 1 ---")
tokens_1 = simple_tokenize(sample_text_1)
print(f"Final Tokens for Example 1: {tokens_1}\n")

sample_text_2 = "Natural Language Processing (NLP) is exciting!"
print("--- Tokenization Example 2 ---")
tokens_2 = simple_tokenize(sample_text_2)
print(f"Final Tokens for Example 2: {tokens_2}\n")

sample_text_3 = "  Leading and trailing spaces. And... multiple    spaces.  "
print("--- Tokenization Example 3 ---")
tokens_3 = simple_tokenize(sample_text_3)
print(f"Final Tokens for Example 3: {tokens_3}\n")
