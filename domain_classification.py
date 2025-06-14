import os
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import ast

def load_domain_embeddings(embedding_dir):
    domain_file = embedding_dir / f"special_terms_{embedding_dir.name.split('_')[0]}.json"
    
    if not domain_file.exists():
        print(f"Warning: Domain file {domain_file} not found")
        return {}
    
    with open(domain_file, 'r', encoding='utf-8') as f:
        domains = json.load(f)
    
    # Convert lists back to numpy arrays for calculations
    for domain, embedding in domains.items():
        domains[domain] = np.array(embedding)
    
    return domains

def load_salient_term_embeddings(embedding_dir):
   
    embeddings = {}
    
    part_name_map = {
        'code': 'code',
        'comment': 'comment',
        'narrative_md': 'markdown narrative',
        'salient_terms': 'all combined'
    }
    
    embed_type = embedding_dir.name.split('_')[0]
    
    for subfolder_path in embedding_dir.glob('*/'):
        notebook_name = subfolder_path.name
        embeddings[notebook_name] = {}
        
        for json_file in subfolder_path.glob('*.json'):
            file_name = json_file.name
            
            if file_name.startswith('salient_terms_'):
                parts = file_name.split('_')
                if len(parts) >= 3:
                    file_part = 'salient_terms'
                    term_type = parts[2]  # tfidf, bow, or lda
                else:
                    continue
            elif file_name.startswith('narrative_md_salient_'):
                parts = file_name.split('_')
                if len(parts) >= 4:
                    file_part = 'narrative_md'
                    term_type = parts[3]  # tfidf, bow, or lda
                else:
                    continue
            elif file_name.startswith('code_salient_') or file_name.startswith('comment_salient_'):
                parts = file_name.split('_')
                if len(parts) >= 3:
                    file_part = parts[0]  # code or comment
                    term_type = parts[2]  # tfidf, bow, or lda
                else:
                    continue
            
            if f"_{embed_type}" in term_type:
                term_type = term_type.split(f"_{embed_type}")[0]
            
            part_label = part_name_map.get(file_part, file_part)
            
            key = (notebook_name, part_label, term_type)
            
            with open(json_file, 'r', encoding='utf-8') as f:
                terms_data = json.load(f)
            
            for term, embedding in terms_data.items():
                terms_data[term] = np.array(embedding)
            
            embeddings[notebook_name][key] = terms_data
            # print(f"Successfully loaded {file_name} as {key}")
    
    return embeddings

def calculate_domain_similarities(term_embeddings, domain_embeddings):
    """
    Calculate cosine similarity between each term and each domain
    
    Args:
        term_embeddings (dict): Dictionary mapping terms to embeddings
        domain_embeddings (dict): Dictionary mapping domains to embeddings
        
    Returns:
        dict: Dictionary mapping domains to their cumulative similarity scores
    """
    if not term_embeddings or not domain_embeddings:
        return {}
    
    domain_similarities = defaultdict(float)
    
    for term, term_embedding in term_embeddings.items():
        for domain, domain_embedding in domain_embeddings.items():
            if np.all(term_embedding == 0) or np.all(domain_embedding == 0):
                continue
            
            term_emb_reshaped = term_embedding.reshape(1, -1)
            domain_emb_reshaped = domain_embedding.reshape(1, -1)
            
            similarity = cosine_similarity(term_emb_reshaped, domain_emb_reshaped)[0][0]
            
            # Add to cumulative score
            domain_similarities[domain] += max(0, similarity)  # Only count positive similarities since negative implies opposite and should not affect other +ve similarities
    
    return domain_similarities

def get_top_domains(domain_similarities, n=10):
  
    sorted_domains = sorted(domain_similarities.items(), key=lambda x: x[1], reverse=True)
    return [domain for domain, _ in sorted_domains[:n]]

def main():
    current_dir = Path.cwd()
    
    embedding_dirs = {
        'bert': current_dir / 'bert_embeddings',
        'word2vec': current_dir / 'word2vec_embeddings',
        'glove': current_dir / 'glove_embeddings'
    }
    
    output_file = current_dir / 'domain_classification_results.csv'
    
    for name, dir_path in embedding_dirs.items():
        if not dir_path.exists():
            print(f"Warning: {name} embeddings directory not found at {dir_path}")
    
    domain_embeddings = {}
    for embed_type, embed_dir in embedding_dirs.items():
        if embed_dir.exists():
            domain_embeddings[embed_type] = load_domain_embeddings(embed_dir)
    
    salient_term_embeddings = {}
    for embed_type, embed_dir in embedding_dirs.items():
        if embed_dir.exists():
            salient_term_embeddings[embed_type] = load_salient_term_embeddings(embed_dir)
    
    results = []

    for embed_type in tqdm(embedding_dirs.keys(), desc="Processing embedding types"):
        if embed_type not in domain_embeddings or embed_type not in salient_term_embeddings:
            continue
        
        domains = domain_embeddings[embed_type]
        
        for subfolder, notebooks in tqdm(salient_term_embeddings[embed_type].items(), 
                                         desc=f"Processing {embed_type} subfolders"):
            
            for (notebook, part, term_type), terms in notebooks.items():
                similarities = calculate_domain_similarities(terms, domains)
                
                top_domains = get_top_domains(similarities, 10)
                
                top_domains_str = str(top_domains)
                
                results.append({
                    'notebook': notebook,
                    'part': part,
                    'term_type': term_type,
                    'embedding_type': embed_type,
                    'top_domains': top_domains_str
                })

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['notebook', 'part', 'term_type', 'embedding_type', 'top_domains']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults written to {output_file}")

if __name__ == "__main__":
    main()