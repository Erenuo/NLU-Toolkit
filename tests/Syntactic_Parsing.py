# This script demonstrates the basic principles of Syntactic Parsing.
# Syntactic parsing (or parsing) is the process of analyzing a string of symbols,
# either in natural language or in computer languages, conforming to the rules
# of a formal grammar. It aims to determine the grammatical structure of a sentence.

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# If you run this for the first time, you might need to download NLTK data:
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker_tab') # For named entity recognition (optional, but good for phrase chunking)
nltk.download('words') # For named entity recognition (optional)

def perform_syntactic_parsing(sentence):
    """
    Performs a simplified step-by-step syntactic parsing on a given sentence.
    This example uses NLTK for tokenization and POS tagging, then
    demonstrates a very basic rule-based chunking.
    """
    print(f"\n--- Analyzing the sentence: '{sentence}' ---")

    # Step 1: Tokenization
    # Break the sentence into individual words or tokens.
    print("Step 1: Tokenization")
    tokens = word_tokenize(sentence)
    print(f"  Tokens: {tokens}\n")

    # Step 2: Part-of-Speech (POS) Tagging
    # Assign a grammatical category (e.g., noun, verb, adjective) to each token.
    # NLTK's pos_tag uses the Penn Treebank tagset by default.
    print("Step 2: Part-of-Speech (POS) Tagging")
    pos_tags = pos_tag(tokens)
    print(f"  POS Tags: {pos_tags}\n")

    # Step 3: Chunking (Simplified Rule-Based Phrase Identification)
    # Group words into "chunks" or phrases (e.g., Noun Phrases, Verb Phrases)
    # based on their POS tags. This is a very basic, illustrative example.
    # A real parser uses more complex grammar rules.
    print("Step 3: Chunking (Simplified Rule-Based Phrase Identification)")
    chunks = []
    i = 0
    while i < len(pos_tags):
        word, tag = pos_tags[i]
        if tag.startswith('NN'): # Noun (NN, NNS, NNP, NNPS)
            # Try to find a Noun Phrase (e.g., Determiner + Adjective(s) + Noun(s))
            np_words = [word]
            np_tags = [tag]
            j = i + 1
            while j < len(pos_tags):
                next_word, next_tag = pos_tags[j]
                if next_tag.startswith('DT') or next_tag.startswith('JJ') or next_tag.startswith('NN'):
                    np_words.append(next_word)
                    np_tags.append(next_tag)
                    j += 1
                else:
                    break
            chunks.append(f"NP: {' '.join(np_words)} ({'/'.join(np_tags)})")
            i = j
        elif tag.startswith('VB'): # Verb (VB, VBD, VBG, VBN, VBP, VBZ)
            # Try to find a Verb Phrase (e.g., Adverb + Verb + Noun Phrase)
            vp_words = [word]
            vp_tags = [tag]
            j = i + 1
            while j < len(pos_tags):
                next_word, next_tag = pos_tags[j]
                if next_tag.startswith('RB') or next_tag.startswith('VB') or next_tag.startswith('NN') or next_tag.startswith('DT') or next_tag.startswith('JJ'):
                    vp_words.append(next_word)
                    vp_tags.append(next_tag)
                    j += 1
                else:
                    break
            chunks.append(f"VP: {' '.join(vp_words)} ({'/'.join(vp_tags)})")
            i = j
        else: # Other parts of speech (e.g., prepositions, conjunctions)
            chunks.append(f"{tag}: {word}")
            i += 1
    print(f"  Identified Chunks (Simplified): {chunks}\n")


    # NLTK's ne_chunk can give a basic tree structure, often highlighting Named Entities
    # which can be seen as a form of phrase chunking.
    try:
        # Requires 'maxent_ne_chunker' and 'words' corpora
        tree = nltk.ne_chunk(pos_tags)
        print("  NLTK's Named Entity Chunking Output:")
        print(tree)
    except Exception as e:
        print(f"  Could not generate NLTK tree (ensure 'maxent_ne_chunker' and 'words' are downloaded): {e}")

    print("\n--- Syntactic Parsing Process Complete ---")


# --- Example Usage ---
sentences_to_parse = [
    "The quick brown fox jumps over the lazy dog.",
    "She quickly ran to the store.",
    "The cute cat meowed at the dragon."
]

print("--- Starting Syntactic Parsing Examples ---")
for s in sentences_to_parse:
    perform_syntactic_parsing(s)
    print("\n" + "="*70 + "\n") # Separator for readability
