import re
from pathlib import Path
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import LdaModel
import nltk
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def preprocess_text(text, file_type):
    """
    Preprocesses the input text by applying file-specific cleaning rules,
    then tokenizing, removing stopwords, and lemmatizing.

    Args:
        text (str): The input text to be processed.
        file_type (str): The type of file ('code', 'comment', 'narrative', 'unknown').

    Returns:
        list: A list of preprocessed tokens (strings).
    """
    cleaned_text = ""
    lines = text.splitlines()
    cleaned_lines = []

    if file_type == 'code':
        for line in lines:
            cleaned_line = re.sub(r'^Content of code cell #\d+:', '', line).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        cleaned_text = ' \n'.join(cleaned_lines)
    elif file_type == 'comment':
        skip_headers = True
        for line in lines:
            if skip_headers and ('Comments extracted from:' in line or '=================' in line):
                continue
            skip_headers = False
            if not line.strip() or '------------------------------' in line or line.startswith('Comments from Code Cell'):
                continue
            cleaned_line = re.sub(r'^\s*-\s*', '', line).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        cleaned_text = ' \n'.join(cleaned_lines)
    elif file_type == 'narrative':
        skip_headers = True
        for line in lines:
            if skip_headers and ('Markdown Narrative from:' in line or '=================' in line):
                continue
            skip_headers = False
            if not line.strip() or '------------------------------' in line or line.startswith('Markdown Cell #'):
                continue
            cleaned_lines.append(line.strip())
        cleaned_text = ' \n'.join(cleaned_lines)
    else: # 'unknown' or other types
        cleaned_text = text # Use original text if no specific rule

    # Tokenization + Stopword Removal + Lemmatization
    result = []
    lemmatizer = WordNetLemmatizer()
    for token in simple_preprocess(cleaned_text):
        if token not in STOPWORDS and len(token) > 3:
            # Lemmatize verbs
            result.append(lemmatizer.lemmatize(token, pos='v'))
    return result


def get_bow_terms(term_freq, limit=30):
    """Gets the most common terms (Descending frequency)."""
    # Sort by frequency in descending 
    sorted_items = sorted(term_freq.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[:limit]


def extract_salient_terms():
    """
    Extracts salient terms using LDA, BOW (most frequent), and TF-IDF.
    Processes files in 'nb_contents', calculates TF-IDF globally,
    and saves detailed results per document in 'salient_terms'.
    """
    current_dir = Path.cwd()
    nb_contents_dir = current_dir / 'nb_contents'
    results_dir = current_dir / 'salient_terms'
    results_dir.mkdir(exist_ok=True)


    # print("Building global corpus for TF-IDF...")
    all_docs_for_tfidf = [] # List of token lists
    all_file_paths_for_tfidf = []

    for txt_file in nb_contents_dir.glob('**/*.txt'):
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f: # Ignore errors for robustness
            text = f.read()
            file_stem = txt_file.stem.lower()
            if 'code' in file_stem: file_type = 'code'
            elif 'comment' in file_stem: file_type = 'comment'
            elif 'narrative' in file_stem: file_type = 'narrative'
            else: file_type = 'unknown'

            processed_tokens = preprocess_text(text, file_type)
            all_docs_for_tfidf.append(processed_tokens) # Store list of tokens
            all_file_paths_for_tfidf.append(txt_file)


    all_docs_strings = [' '.join(doc) for doc in all_docs_for_tfidf]
    # print(f"Fitting TF-IDF Vectorizer on {len(all_docs_strings)} documents...")
    tfidf_vectorizer = None
    tfidf_feature_names = []
    # try:
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.95, min_df=2)
    tfidf_vectorizer.fit(all_docs_strings) # Fit on the global corpus strings
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        # print("TF-IDF Vectorizer fitted.")
    # except ValueError as e:
         # print(f"Error fitting TF-IDF Vectorizer: {e}. Ensure corpus is not empty after preprocessing.")

    # except Exception as e:
         # print(f"Unexpected error fitting TF-IDF Vectorizer: {e}")


    subfolder_files = defaultdict(list)
    for txt_file in nb_contents_dir.glob('**/*.txt'):
         if txt_file.name.startswith('.'): continue
         subfolder = txt_file.parent.relative_to(nb_contents_dir)
         subfolder_files[subfolder].append(txt_file)

    for subfolder, files in subfolder_files.items():
        # print(f"\nProcessing subfolder: {subfolder} ({len(files)} files)")

        result_subfolder = results_dir / subfolder
        result_subfolder.mkdir(exist_ok=True, parents=True)

        documents = [] # List of token lists for this subfolder
        file_paths = [] # Corresponding file paths
        doc_strings_current_subfolder = [] # List of joined token strings

        for txt_file in files:
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    file_stem = txt_file.stem.lower()
                    if 'code' in file_stem: file_type = 'code'
                    elif 'comment' in file_stem: file_type = 'comment'
                    elif 'narrative' in file_stem: file_type = 'narrative'
                    else: file_type = 'unknown'

                    processed_tokens = preprocess_text(text, file_type)
                    documents.append(processed_tokens)
                    doc_strings_current_subfolder.append(' '.join(processed_tokens))
                    file_paths.append(txt_file)
            except Exception as e:
                 # print(f"  Warning: Could not process {txt_file.name} in subfolder {subfolder}: {e}")
                 continue

        if not documents:
             # print(f"  Skipping subfolder {subfolder}: No documents processed successfully.")
             continue

        # print(f"  Running LDA for subfolder {subfolder}...")
        lda_model = None
        dictionary = None
        corpus = None
        try:
            dictionary = corpora.Dictionary(documents)
            corpus = [dictionary.doc2bow(doc) for doc in documents]

            if not dictionary or not corpus or not any(corpus): 
                 # print(f"  Skipping LDA for {subfolder}: Empty dictionary or corpus after preprocessing/filtering.")
                 continue
            else:
                # Adjust num_topics based on dictionary size or document count
                num_topics = min(10, max(1, len(dictionary) // 5, len(documents) // 2)) 
                num_topics = max(1, num_topics) # Ensure at least 1

                # print(f"    Using {num_topics} topics for LDA.")
                lda_model = LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=num_topics,
                    random_state=42,
                    passes=15, # Increased passes slightly
                    alpha='auto',
                    eta='auto', # symmetric eta often works well
                    per_word_topics=True
                )
        except Exception as e:
             print(f"  Error running LDA for {subfolder}: {e}")


        subfolder_tfidf_matrix = None
        if tfidf_vectorizer: 
            # print(f"  Calculating TF-IDF for subfolder {subfolder}...")
            try:
                subfolder_tfidf_matrix = tfidf_vectorizer.transform(doc_strings_current_subfolder)
            except Exception as e:
                print(f"  Error transforming documents for TF-IDF in {subfolder}: {e}")


       
        # print(f"  Generating salient term files for {subfolder}...")
        for i, (txt_file, doc_tokens) in enumerate(zip(file_paths, documents)):
            result_file = result_subfolder / f"{txt_file.stem}_salient.txt"

           
            term_freq = defaultdict(int)
            for term in doc_tokens:
                term_freq[term] += 1

            bow_terms_most_freq = get_bow_terms(term_freq, limit=30)

            sorted_lda_terms = []
            if lda_model and corpus and i < len(corpus): # Check prerequisites
                try:
                    doc_bow = corpus[i]
                    if not doc_bow: # Handle empty documents after bow conversion
                         print(f"    Skipping LDA for {txt_file.name}: Document empty after preprocessing.")
                    else:
                        # Get topic distribution, weighted by probability
                        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.01) # Threshold low probability topics
                        doc_salient_terms_lda = defaultdict(float)
                        num_significant_topics = 0
                        for topic_id, prob in doc_topics:
                            if prob > 0.05: # Consider topic significant if prob > 5%
                                num_significant_topics+=1
                                topic_terms = lda_model.get_topic_terms(topic_id, topn=20) # Get top terms for this topic_id
                                for term_id, score in topic_terms:
                                    term = dictionary[term_id] # Convert ID to word
                                    doc_salient_terms_lda[term] += score * prob # Weight by topic probability

                        if not doc_salient_terms_lda:
                             # Fallback: If no significant topics, take top terms from highest prob topic
                            if doc_topics:
                                top_topic_id, _ = max(doc_topics, key=lambda item: item[1])
                                topic_terms = lda_model.get_topic_terms(top_topic_id, topn=30)
                                for term_id, score in topic_terms:
                                     doc_salient_terms_lda[dictionary[term_id]] = score # Use raw score as fallback

                        sorted_lda_terms = sorted(doc_salient_terms_lda.items(), key=lambda x: x[1], reverse=True)[:30]
                except Exception as e:
                     print(f"    Error getting LDA terms for {txt_file.name}: {e}")

            sorted_tfidf_terms = []
            if subfolder_tfidf_matrix is not None and i < subfolder_tfidf_matrix.shape[0]:
                try:
                    doc_tfidf_scores = {}
                    vector_slice = subfolder_tfidf_matrix[i] # Row for the current doc
                    for col_idx in vector_slice.nonzero()[1]:
                        term = tfidf_feature_names[col_idx]
                        score = vector_slice[0, col_idx]
                        doc_tfidf_scores[term] = score
                    # Sort by score and take top 30
                    sorted_tfidf_terms = sorted(doc_tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:30]
                except IndexError as e:
                     print(f"    Index Error getting TF-IDF terms for {txt_file.name} (likely feature name mismatch): {e}")
                except Exception as e:
                     print(f"    Error getting TF-IDF terms for {txt_file.name}: {e}")


            try:
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f"Salient Terms Analysis for: {txt_file.name}\n")
                    f.write("=" * 60 + "\n\n")

                    f.write("Top 30 LDA Salient Terms (Weighted by Topic Probability):\n")
                    f.write("-" * 60 + "\n")
                    if sorted_lda_terms:
                        for term, score in sorted_lda_terms:
                            f.write(f"- {term}: {score:.4f}\n")
                    else:
                        f.write("  N/A (LDA processing failed or no salient terms found)\n")
                    f.write("\n\n")

                    f.write("Top 30 BOW Terms (Most Frequent):\n")
                    f.write("-" * 60 + "\n")
                    if bow_terms_most_freq:
                        for term, freq in bow_terms_most_freq:
                            f.write(f"- {term}: {freq}\n")
                    else:
                        f.write("  N/A (No terms found after preprocessing)\n")
                    f.write("\n\n")

                    f.write("Top 30 TF-IDF Salient Terms:\n")
                    f.write("-" * 60 + "\n")
                    if sorted_tfidf_terms:
                        for term, score in sorted_tfidf_terms:
                            f.write(f"- {term}: {score:.4f}\n")
                    else:
                        f.write("  N/A (TF-IDF processing failed or no terms found)\n")
                    f.write("\n") # Single newline at the end

            except Exception as e:
                print(f"  Error writing salient terms file {result_file.name}: {e}")

        print(f"  Generating overall salient_terms.txt for {subfolder} by analyzing combined content...")
        try:
            all_subfolder_tokens = [token for doc in documents for token in doc]

            if not all_subfolder_tokens:
                # print(f"    Skipping overall analysis for {subfolder}: No tokens found after preprocessing all files.")
                # Create an empty or minimal salient_terms.txt file
                with open(result_subfolder / 'salient_terms.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Overall Salient Terms Analysis for Subfolder: {subfolder}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("No processable content found in this subfolder.\n")
                continue # Skip to the next subfolder

            subfolder_term_freq = defaultdict(int)
            for term in all_subfolder_tokens:
                subfolder_term_freq[term] += 1
            sorted_overall_bow_least_freq = get_bow_terms(subfolder_term_freq, limit=30)

            sorted_overall_lda = []
            try:
                subfolder_dict = corpora.Dictionary([all_subfolder_tokens])
                subfolder_corpus_bow = [subfolder_dict.doc2bow(all_subfolder_tokens)] # List with one BoW vector

                if not subfolder_dict or not subfolder_corpus_bow or not subfolder_corpus_bow[0]:
                    print(f"    Skipping overall LDA for {subfolder}: Empty dictionary/corpus for combined document.")
                else:
                    num_combined_topics = min(5, max(1, len(subfolder_dict) // 10)) # Heuristic for single doc
                    num_combined_topics = max(1, num_combined_topics) # Ensure at least 1 topic
                    print(f"    Running new LDA with {num_combined_topics} topics on combined content of {subfolder}...")
                    subfolder_lda_model = LdaModel(
                        corpus=subfolder_corpus_bow,
                        id2word=subfolder_dict,
                        num_topics=num_combined_topics,
                        random_state=42,
                        passes=15,
                        alpha='auto',
                        eta='auto'
                    )
                    overall_lda_scores = defaultdict(float)
                    for topic_id in range(subfolder_lda_model.num_topics):
                        topic_terms = subfolder_lda_model.get_topic_terms(topic_id, topn=30) # Get terms for this topic
                        for term_id, score in topic_terms:
                            term = subfolder_dict[term_id]
                            overall_lda_scores[term] = max(overall_lda_scores[term], score) # Keep highest score
                    sorted_overall_lda = sorted(overall_lda_scores.items(), key=lambda x: x[1], reverse=True)[:30]

            except Exception as e:
                print(f"    Error running new LDA on combined content for {subfolder}: {e}")

            sorted_overall_tfidf = []
            if tfidf_vectorizer: # Check if global vectorizer exists
                try:
                    subfolder_combined_string = ' '.join(all_subfolder_tokens)
                    combined_tfidf_vector = tfidf_vectorizer.transform([subfolder_combined_string])

                    combined_tfidf_scores = {}
                    rows, cols = combined_tfidf_vector.nonzero()
                    for _, col in zip(rows, cols): # Row index will always be 0 here
                         if col < len(tfidf_feature_names):
                            term = tfidf_feature_names[col]
                            score = combined_tfidf_vector[0, col]
                            combined_tfidf_scores[term] = score
                         else:
                             print(f"    Warning: TF-IDF column index {col} out of bounds for feature names.")

                    sorted_overall_tfidf = sorted(combined_tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:30]
                except Exception as e:
                    print(f"    Error calculating TF-IDF on combined content for {subfolder}: {e}")

            output_summary_file = result_subfolder / 'salient_terms.txt'
            with open(output_summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Overall Salient Terms Analysis for Subfolder: {subfolder}\n")
                f.write("=" * 60 + "\n\n")

                f.write("Top 30 Overall LDA Salient Terms (Combined Content Run):\n")
                f.write("-" * 60 + "\n")
                if sorted_overall_lda:
                    for term, score in sorted_overall_lda:
                        f.write(f"- {term}: {score:.4f}\n")
                else:
                    f.write("  N/A (LDA analysis on combined content failed or no terms found)\n")
                f.write("\n\n")

                f.write("Top 30 Overall BOW Terms (Least Frequent - Combined Content):\n")
                f.write("-" * 60 + "\n")
                if sorted_overall_bow_least_freq:
                    for term, freq in sorted_overall_bow_least_freq:
                        f.write(f"- {term}: {freq}\n")
                else:
                    f.write("  N/A (No terms found after preprocessing)\n")
                f.write("\n\n")

                f.write("Top 30 Overall TF-IDF Salient Terms (Combined Content):\n")
                f.write("-" * 60 + "\n")
                if sorted_overall_tfidf:
                    for term, score in sorted_overall_tfidf:
                        f.write(f"- {term}: {score:.4f}\n")
                else:
                    f.write("  N/A (TF-IDF analysis on combined content failed or no terms found)\n")
                f.write("\n") # Single newline at the end

        except Exception as e:
            print(f"  Error during overall analysis or writing salient_terms.txt for {subfolder}: {e}")

    print(f"\nSalient terms extraction complete. Results saved to '{results_dir}'")


if __name__ == "__main__":
    extract_salient_terms() 