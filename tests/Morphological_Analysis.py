# This script demonstrates the basic principles of Morphological Analysis.
# Morphological analysis is the process of breaking down words into their
# smallest meaningful units (morphemes) and identifying their grammatical
# properties.

def perform_morphological_analysis(word):
    """
    Performs a simplified step-by-step morphological analysis on a given word.
    """
    print(f"\n--- Analyzing the word: '{word}' ---")

    # Step 1: Initial Word
    # The word as it is presented.
    print("Step 1: Initial Word")
    print(f"  Word: '{word}'\n")

    # Step 2: Lowercasing (Standard Preprocessing)
    # Convert the word to lowercase to ensure consistency and
    # treat "Run" and "run" as the same base form.
    print("Step 2: Lowercasing")
    processed_word = word.lower()
    print(f"  Lowercased Word: '{processed_word}'\n")

    # Step 3: Identify Potential Prefixes (Simplified)
    # A prefix is a morpheme added to the beginning of a word to modify its meaning.
    # This is a very limited list for demonstration.
    prefixes = ["un", "re", "pre", "dis", "anti"]
    identified_prefix = ""
    for prefix in prefixes:
        if processed_word.startswith(prefix) and len(processed_word) > len(prefix):
            identified_prefix = prefix
            processed_word = processed_word[len(prefix):] # Remove prefix for further analysis
            print(f"Step 3: Identified Prefix")
            print(f"  Prefix: '{identified_prefix}'")
            print(f"  Remaining Word (after prefix removal): '{processed_word}'\n")
            break # Assume only one prefix for simplicity

    # Step 4: Identify Potential Suffixes (Simplified)
    # A suffix is a morpheme added to the end of a word to modify its meaning
    # or change its grammatical function (e.g., noun to adjective).
    # This list includes both inflectional and derivational suffixes for demo.
    suffixes = {
        "ing": "Verb (present participle) / Gerund",
        "ed": "Verb (past tense) / Adjective",
        "s": "Noun (plural) / Verb (3rd person singular)",
        "ly": "Adverb",
        "tion": "Noun",
        "able": "Adjective",
        "er": "Noun (agent) / Adjective (comparative)",
        "est": "Adjective (superlative)",
        "ness": "Noun",
        "ful": "Adjective"
    }
    identified_suffix = ""
    suffix_meaning = ""
    original_processed_word_for_suffix_check = processed_word # Store before potential modification

    # Sort suffixes by length in descending order to match longest possible suffix first
    sorted_suffixes = sorted(suffixes.keys(), key=len, reverse=True)

    for suffix in sorted_suffixes:
        if processed_word.endswith(suffix) and len(processed_word) > len(suffix):
            identified_suffix = suffix
            suffix_meaning = suffixes[suffix]
            # For demonstration, we'll keep the "stem" as the word without the suffix
            processed_word = processed_word[:-len(suffix)]
            print(f"Step 4: Identified Suffix")
            print(f"  Suffix: '{identified_suffix}'")
            print(f"  Suffix's Common Function/Meaning: '{suffix_meaning}'")
            print(f"  Potential Stem (after suffix removal): '{processed_word}'\n")
            break # Assume only one suffix for simplicity

    # Step 5: Identify Potential Root/Stem (Simplified)
    # The root is the core morpheme of a word, carrying its main lexical meaning.
    # In a real system, this would involve a lexicon lookup. Here, it's what's left.
    print("Step 5: Potential Root/Stem")
    root = processed_word
    print(f"  Derived Root/Stem: '{root}'\n")

    # Step 6: Infer Grammatical Information (Simplified)
    # Based on the identified morphemes, infer part-of-speech or other properties.
    # This is a very basic inference, not a robust POS tagger.
    inferred_pos = "Unknown"
    if identified_suffix:
        inferred_pos = suffix_meaning.split(' ')[0] # Take the first part of the meaning
    elif identified_prefix:
        # Prefixes usually don't change POS, so try to guess based on root
        # (This part is highly speculative without a lexicon)
        if root.endswith('e'): # Very basic guess for verbs
            inferred_pos = "Verb"
        elif root.endswith('y'): # Very basic guess for nouns/adjectives
            inferred_pos = "Noun/Adjective"
        else:
            inferred_pos = "Base Form (likely Noun/Verb/Adjective)"
    else:
        # If no affixes, it's likely a base form
        inferred_pos = "Base Form (Noun/Verb/Adjective)"

    print("Step 6: Inferred Grammatical Information")
    print(f"  Inferred Part-of-Speech/Type: {inferred_pos}\n")

    print("--- Morphological Analysis Summary ---")
    print(f"  Original Word: '{word}'")
    print(f"  Prefix: '{identified_prefix if identified_prefix else 'None'}'")
    print(f"  Root/Stem: '{root}'")
    print(f"  Suffix: '{identified_suffix if identified_suffix else 'None'}'")
    if identified_suffix:
        print(f"  Suffix Meaning/Function: '{suffix_meaning}'")
    print(f"  Inferred Grammatical Role: {inferred_pos}")


# --- Example Usage ---
words_to_analyze = [
    "unhappiness",
    "restarted",
    "running",
    "quickly",
    "dogs",
    "beautiful",
    "precaution",
    "anti-establishment", # Note: hyphenated words are tricky for simple tokenizers
    "jumped",
    "friendship"
]

print("--- Starting Morphological Analysis Examples ---")
for w in words_to_analyze:
    perform_morphological_analysis(w)
    print("\n" + "="*50 + "\n") # Separator for readability
