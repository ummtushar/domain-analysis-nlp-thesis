import os
import json
import csv
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm

def extract_kaggle_tags(data_dir="data"):
    """
    Extracts "keywords" from all kernel-metadata.json files in the nested folder structure:
    data/username/notebook-name/kernel-metadata.json
    
    Args:
        data_dir (str): Path to the data directory (default is "data" in current directory)
        
    Returns:
        dict: A dictionary mapping notebook names to their associated tags
        list: A list of all unique tags
    """
    notebook_tags = {}
    all_tags = []
    
    
    user_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Found {len(user_folders)} user folders")
    
    for user in tqdm(user_folders, desc="Processing user folders"):
        user_path = os.path.join(data_dir, user)
        
        notebook_folders = [f for f in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, f))]
        
        for notebook in notebook_folders:
            notebook_path = os.path.join(user_path, notebook)
            json_path = os.path.join(notebook_path, "kernel-metadata.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if "keywords" in metadata and metadata["keywords"]:
                        keywords = metadata["keywords"]
                        notebook_tags[notebook] = keywords
                        all_tags.extend(keywords)
                    else:
                        # Empty list for notebooks without keywords
                        notebook_tags[notebook] = []
                except Exception as e:
                    print(f"Error processing {json_path}: {e}")
                    notebook_tags[notebook] = []
    
    unique_tags = list(set(all_tags))
    
    # print(f"\nExtracted {len(all_tags)} tags in total")
    # print(f"Found {len(unique_tags)} unique tags")
    
    return notebook_tags, unique_tags

def extract_top_domains_from_csv(csv_path):
    """
    Extracts top domains from the domain classification CSV file
    
    Args:
        csv_path (Path): Path to the domain classification results CSV
        
    Returns:
        dict: A nested dictionary mapping notebook, part, term_type, and embedding_type to top domains
    """
    
    
    df = pd.read_csv(csv_path)
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for _, row in df.iterrows():
        notebook = row['notebook']
        part = row['part']
        term_type = row['term_type']
        embedding_type = row['embedding_type']
        
        try:
            top_domains_str = row['top_domains']
            if isinstance(top_domains_str, str):
                try:
                    # This should handle strings like "['domain1', 'domain2', ...]"
                    top_domains = ast.literal_eval(top_domains_str)
                except (ValueError, SyntaxError):
                    # If that fails, try a simpler regex-based approach
                    matches = re.findall(r"'(.*?)'", top_domains_str)
                    if matches:
                        top_domains = matches
                    else:
                        print(f"Warning: Could not parse top_domains: {top_domains_str}")
                        top_domains = []
            else:
                print(f"Warning: top_domains is not a string: {top_domains_str}")
                top_domains = []
        except Exception as e:
            print(f"Error parsing top_domains for {notebook}, {part}, {term_type}, {embedding_type}: {e}")
            top_domains = []
        
        # Store in nested dictionary
        results[notebook][part][term_type][embedding_type] = top_domains
    
    return results

def calculate_precision_at_k(predicted_domains, ground_truth_tags, k=3):
   

    if not ground_truth_tags or len(ground_truth_tags) == 0 or len(ground_truth_tags)<k:
        # If no ground truth tags, we cannot calculate P@k
        
        return None
    
    if not predicted_domains or len(predicted_domains) == 0:
        # If no predictions, precision is 0
        return 0.0

    
    # Limit to top k predictions
    predicted_domains = predicted_domains[:k]
    
    # Convert both lists to lowercase for case-insensitive comparison
    predicted_lower = [d.lower() if isinstance(d, str) else str(d).lower() for d in predicted_domains]
    truth_lower = [t.lower() if isinstance(t, str) else str(t).lower() for t in ground_truth_tags]
    
    # Count correct predictions
    correct = sum(1 for domain in predicted_lower if domain in truth_lower)
    
    # Calculate precision
    precision = correct / min(k, len(predicted_domains))
    
    return precision


def write_all_unique_tags(tags_list, output_file="all_unique_tags.txt"):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(set(tags_list)))
    
    print(f"Wrote {len(tags_list)} unique tags to {output_file}")

def main():
    current_dir = Path.cwd()
    
    csv_path = current_dir / 'domain_classification_results.csv'
    
    all_unique_tags_file = current_dir / 'all_unique_tags.txt'
    
    notebook_tags, all_unique_tags = extract_kaggle_tags()
        
    write_all_unique_tags(all_unique_tags, all_unique_tags_file)
    
    print("\nSample of notebook tags:")
    sample_count = 0
    for notebook, tags in notebook_tags.items():
        print(f"{notebook}: {tags}")
        sample_count += 1
        if sample_count >= 3:
            break
    
    try:
        predicted_domains = extract_top_domains_from_csv(csv_path)
    except FileNotFoundError as e:
        print(e)
        return
    
    print("\nSample of predicted domains:")
    sample_count = 0
    for notebook, parts in predicted_domains.items():
        for part, term_types in parts.items():
            for term_type, embedding_types in term_types.items():
                for embedding_type, domains in embedding_types.items():
                    print(f"{notebook}, {part}, {term_type}, {embedding_type}: {domains}")
                    sample_count += 1
                    if sample_count >= 3:
                        break
                if sample_count >= 3:
                    break
            if sample_count >= 3:
                break
        if sample_count >= 3:
            break
    
    results = []
    
    missing_notebooks = []
    for notebook in predicted_domains.keys():
        if notebook not in notebook_tags and "special_tags" not in notebook_tags:
            missing_notebooks.append(notebook)
    
    if missing_notebooks:
        print(f"\nWarning: {len(missing_notebooks)} notebooks in predictions not found in ground truth tags.")
        print(f"First 5 missing notebooks: {missing_notebooks[:5]}")
    
    all_tags = set()
    for tags in notebook_tags.values():
        all_tags.update(tags)
    
    use_special_tags = "special_tags" in notebook_tags

    count = 0
    
    for notebook, parts in tqdm(predicted_domains.items(), desc="Calculating P@k"):
        if use_special_tags:
            ground_truth = notebook_tags.get("special_tags", [])
        else:
            ground_truth = notebook_tags.get(notebook, [])
            
        if not ground_truth and notebook not in missing_notebooks[:5]:
            print(f"Warning: No ground truth tags found for notebook: {notebook}")
        
        for part, term_types in parts.items():
            for term_type, embedding_types in term_types.items():
                for embedding_type, domains in embedding_types.items():
                    p_at_k = calculate_precision_at_k(domains, ground_truth, k=3)

                    if p_at_k is None: count += 1
                    
                    results.append({
                        'notebook': notebook,
                        'part': part,
                        'term_type': term_type,
                        'embedding_type': embedding_type,
                        'p_at_k': p_at_k if p_at_k is not None else np.nan
                    })
    
    output_path = current_dir / 'precision_at_k_results_new.csv'
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['notebook', 'part', 'term_type', 'embedding_type', 'p_at_k']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    df_results = pd.DataFrame(results)
    
    print("\nAverage P@k by embedding type:")
    print(df_results.groupby('embedding_type')['p_at_k'].mean())
    
    print("\nAverage P@k by term type:")
    print(df_results.groupby('term_type')['p_at_k'].mean())
    
    print("\nAverage P@k by part:")
    print(df_results.groupby('part')['p_at_k'].mean())
    
    print("\nAverage P@k by embedding type and term type:")
    print(df_results.groupby(['embedding_type', 'term_type'])['p_at_k'].mean())
    
    print("\nTop 5 best combinations:")
    top_combs = df_results.groupby(['part', 'term_type', 'embedding_type'])['p_at_k'].mean().sort_values(ascending=False).head(5)
    print(top_combs)
    
    print(f"\nResults written to {output_path}")

    print(count)

main()