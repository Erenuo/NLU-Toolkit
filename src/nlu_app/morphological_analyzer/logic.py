from typing import Dict, List

# --- Module-Level Constants ---
# These are the rules for our simplified analyzer.
# They are defined once when the module is imported.

PREFIXES = ["un", "re", "pre", "dis", "anti"]

SUFFIXES = {
    "ing": "Verb (present participle) / Gerund",
    "ed": "Verb (past tense) / Adjective",
    "s": "Noun (plural) / Verb (3rd person singular)",
    "ly": "Adverb",
    "tion": "Noun",
    "able": "Adjective",
    "er": "Noun (agent) / Adjective (comparative)",
    "est": "Adjective (superlative)",
    "ness": "Noun",
    "ful": "Adjective",
    "ship": "Noun" # Added from your example 'friendship'
}
# Sort suffixes by length (desc) to match longest possible suffix first (e.g., 'able' before 'le')
SORTED_SUFFIX_KEYS = sorted(SUFFIXES.keys(), key=len, reverse=True)


def analyze_morphology(word: str) -> Dict[str, str]:
    """
    Performs a simplified morphological analysis on a single word.

    Args:
        word (str): The word to analyze.

    Returns:
        Dict[str, str]: A dictionary containing the analysis results.
    """
    # Handle empty input
    if not word or not word.strip():
        return {
            "original_word": "", "prefix": None, "root": "", "suffix": None,
            "suffix_function": None, "inferred_pos": "Unknown"
        }
        
    original_word = word
    processed_word = word.lower().strip()

    # --- Analysis Steps ---
    
    # 1. Identify Prefix
    identified_prefix = None
    for prefix in PREFIXES:
        if processed_word.startswith(prefix) and len(processed_word) > len(prefix):
            identified_prefix = prefix
            processed_word = processed_word[len(prefix):] # Remove prefix
            break

    # 2. Identify Suffix
    identified_suffix = None
    suffix_function = None
    for suffix in SORTED_SUFFIX_KEYS:
        if processed_word.endswith(suffix) and len(processed_word) > len(suffix):
            identified_suffix = suffix
            suffix_function = SUFFIXES[suffix]
            processed_word = processed_word[:-len(suffix)] # Remove suffix
            break
            
    # 3. The remainder is the potential root/stem
    root = processed_word

    # 4. Infer Part-of-Speech based on findings
    inferred_pos = "Unknown"
    if identified_suffix:
        # Infer POS from the suffix's primary function
        inferred_pos = suffix_function.split(' ')[0]
    else:
        # Basic guess if no suffix is found
        inferred_pos = "Base Form (Noun/Verb/Adjective)"

    # --- Compile Results ---
    analysis_result = {
        "original_word": original_word,
        "prefix": identified_prefix,
        "root": root,
        "suffix": identified_suffix,
        "suffix_function": suffix_function,
        "inferred_pos": inferred_pos
    }
    
    return analysis_result

def analyze_word_list(words: List[str]) -> List[Dict[str, str]]:
    """
    Args:
        words (List[str]): The list of words to analyze.

    Returns:
        List[Dict[str, str]]: A list of analysis result dictionaries.
    """
    return [analyze_morphology(word) for word in words]
