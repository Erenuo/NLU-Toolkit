import math
from collections import Counter

# This script demonstrates the step-by-step calculation of TF-IDF and
# then uses the TF-IDF scores to calculate cosine similarity between documents.
# TF-IDF is a numerical statistic that is intended to reflect how important
# a word is to a document in a collection or corpus.

def tokenize(text):
    """
    A simple tokenizer for TF-IDF demonstration.
    Converts to lowercase, removes punctuation, and splits by whitespace.
    """
    text = text.lower()
    punctuation_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuation_to_remove:
        text = text.replace(char, "")
    return text.split()
def tokenize(text):
    """
    A simple tokenizer for TF-IDF demonstration.
    Converts to lowercase, removes punctuation, and splits by whitespace.
    """
    text = text.lower()
    punctuation_to_remove = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuation_to_remove:
        text = text.replace(char, "")
    return text.split()

def calculate_tf(document_tokens):
    """
    Calculates the Term Frequency (TF) for each term in a document.
    TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
    """
    term_counts = Counter(document_tokens)
    total_terms = len(document_tokens)
    tf_scores = {term: count / total_terms for term, count in term_counts.items()}
    return tf_scores

def calculate_idf(corpus):
    """
    Calculates the Inverse Document Frequency (IDF) for each term across the corpus.
    IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
    """
    num_documents = len(corpus)
    # Create a set of unique terms present in each document
    document_term_sets = [set(tokenize(doc)) for doc in corpus]

    # Count how many documents each term appears in
    term_document_counts = Counter()
    for doc_terms in document_term_sets:
        for term in doc_terms:
            term_document_counts[term] += 1

    idf_scores = {}
    for term, doc_count in term_document_counts.items():
        # Add 1 to the denominator to avoid division by zero for terms not in the corpus (smooth IDF)
        idf_scores[term] = math.log(num_documents / (doc_count + 1)) + 1 # Adding 1 to log result for smooth IDF
    return idf_scores

def calculate_cosine_similarity(vec1, vec2, doc1_name="Document 1", doc2_name="Document 2"):
    """
    Calculates the cosine similarity between two TF-IDF vectors (dictionaries)
    and prints the intermediate calculation steps.
    Cosine Similarity = (Dot Product of A and B) / (Magnitude of A * Magnitude of B)
    """
    print(f"\n    Calculating Cosine Similarity for {doc1_name} and {doc2_name}:")

    # Get all unique terms from both vectors
    all_terms = set(vec1.keys()).union(set(vec2.keys()))
    print(f"      All unique terms involved: {list(all_terms)}")

    # Calculate dot product
    dot_product_terms = []
    for term in all_terms:
        val1 = vec1.get(term, 0)
        val2 = vec2.get(term, 0)
        dot_product_terms.append(f"({val1:.4f} * {val2:.4f})")
    print(f"      Dot Product Calculation: {' + '.join(dot_product_terms)}")
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
    print(f"      Dot Product: {dot_product:.4f}")

    # Calculate magnitudes (Euclidean norm)
    magnitude1_terms = []
    for score in vec1.values():
        magnitude1_terms.append(f"({score:.4f}^2)")
    print(f"      Magnitude of {doc1_name} Calculation: sqrt({' + '.join(magnitude1_terms)})")
    magnitude1 = math.sqrt(sum(score**2 for score in vec1.values()))
    print(f"      Magnitude of {doc1_name}: {magnitude1:.4f}")

    magnitude2_terms = []
    for score in vec2.values():
        magnitude2_terms.append(f"({score:.4f}^2)")
    print(f"      Magnitude of {doc2_name} Calculation: sqrt({' + '.join(magnitude2_terms)})")
    magnitude2 = math.sqrt(sum(score**2 for score in vec2.values()))
    print(f"      Magnitude of {doc2_name}: {magnitude2:.4f}")


    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        print("      One or both magnitudes are zero, cosine similarity is 0.0")
        return 0.0 # If either vector is zero, similarity is zero

    similarity = dot_product / (magnitude1 * magnitude2)
    print(f"      Cosine Similarity = {dot_product:.4f} / ({magnitude1:.4f} * {magnitude2:.4f}) = {similarity:.4f}")
    return similarity

def calculate_tfidf(corpus):
    """
    Calculates the TF-IDF scores for all terms in all documents in the corpus
    and then computes cosine similarity between documents.
    TF-IDF(t,d) = TF(t,d) * IDF(t)
    """
    print("--- TF-IDF Calculation Process ---")

    # Step 1: Tokenize each document in the corpus
    print("\nStep 1: Tokenizing Documents")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    for i, tokens in enumerate(tokenized_corpus):
        print(f"  Document {i+1} Tokens: {tokens}")

    # Step 2: Calculate IDF scores for all unique terms in the corpus
    print("\nStep 2: Calculating Inverse Document Frequency (IDF)")
    idf_scores = calculate_idf(corpus)
    for term, score in idf_scores.items():
        print(f"  IDF('{term}'): {score:.4f}")

    # Step 3: Calculate TF-IDF for each document
    print("\nStep 3: Calculating Term Frequency (TF) and TF-IDF for each document")
    tfidf_results = []
    for i, doc_tokens in enumerate(tokenized_corpus):
        print(f"\n  --- Document {i+1} ---")
        tf_scores = calculate_tf(doc_tokens)
        print(f"    Term Frequencies (TF): {tf_scores}")

        doc_tfidf_scores = {}
        for term, tf_score in tf_scores.items():
            # Get IDF score; if term not in IDF (shouldn't happen with this setup), use a default (e.g., 0)
            idf_score = idf_scores.get(term, 0)
            tfidf_score = tf_score * idf_score
            doc_tfidf_scores[term] = tfidf_score
            print(f"      TF-IDF('{term}') = TF({tf_score:.4f}) * IDF({idf_score:.4f}) = {tfidf_score:.4f}")
        tfidf_results.append(doc_tfidf_scores)

    print("\nTF-IDF Calculation Complete!")

    # Step 4: Calculate Cosine Similarity between documents
    print("\nStep 4: Calculating Cosine Similarity between Documents")
    num_docs = len(tfidf_results)
    for i in range(num_docs):
        for j in range(i + 1, num_docs): # Compare each document with every other document once
            doc1_name = f"Document {i+1}"
            doc2_name = f"Document {j+1}"
            similarity = calculate_cosine_similarity(tfidf_results[i], tfidf_results[j], doc1_name, doc2_name)
            print(f"  Final Cosine Similarity between {doc1_name} and {doc2_name}: {similarity:.4f}")

    return tfidf_results

# --- Example Usage ---
# A small corpus of documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Never jump over the lazy dog.",
    "The dog is brown and lazy.",
    "A quick fox is a fast animal."
]

print("Corpus:")
for i, doc in enumerate(documents):
    print(f"  Document {i+1}: \"{doc}\"")

tfidf_scores_per_document = calculate_tfidf(documents)
