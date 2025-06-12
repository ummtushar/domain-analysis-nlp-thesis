#This code uses multiple threads to process the notebooks in parallel.
#It uses the GROQ API to evaluate the relevance of the domain classifications.
#It saves the results to a CSV file.
import pandas as pd
import os
import re
import ast
import time
import random
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
import threading
import queue
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import salient_terms
    preprocess_text = salient_terms.preprocess_text
except ImportError:
    def preprocess_text(text, file_type):
        if file_type == 'code':
            lines = [re.sub(r'^Content of code cell #\d+:', '', line).strip() 
                     for line in text.splitlines() if line.strip()]
            return '\n'.join(lines)
        elif file_type == 'narrative':
            lines = []
            skip_headers = True
            for line in text.splitlines():
                if skip_headers and ('Markdown Narrative from:' in line or '=' * 10 in line):
                    continue
                skip_headers = False
                if line.strip() and not ('-' * 10 in line) and not line.startswith('Markdown Cell #'):
                    lines.append(line.strip())
            return '\n'.join(lines)
        else:
            return text

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set. Please add it to your .env file.")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)


MAX_WORKERS = 5  
TPM_LIMIT = 200000 
RPM_LIMIT = 1000 


REQUEST_INTERVAL = 60.0 / RPM_LIMIT  # Time in seconds between requests

# Token tracking to respect TPM limits
token_usage_lock = threading.Lock()
token_usage_window = []  # List of (timestamp, token_count) tuples

def track_token_usage(timestamp, token_count):
    """Track token usage within the current minute window"""
    with token_usage_lock:
        # Remove entries older than 60 seconds
        current_time = time.time()
        token_usage_window[:] = [(ts, tc) for ts, tc in token_usage_window if current_time - ts < 60]
        
        # Add new entry
        token_usage_window.append((timestamp, token_count))
        
        # Calculate current usage
        current_usage = sum(tc for _, tc in token_usage_window)
        return current_usage

def wait_for_token_capacity(estimated_tokens):
    """Wait until there's enough token capacity to process the request"""
    while True:
        with token_usage_lock:
            # Clean up old entries
            current_time = time.time()
            token_usage_window[:] = [(ts, tc) for ts, tc in token_usage_window if current_time - ts < 60]
            
            # Calculate current usage
            current_usage = sum(tc for _, tc in token_usage_window)
            
            # Check if we have capacity
            if current_usage + estimated_tokens <= TPM_LIMIT:
                return
        
        # If we don't have capacity, wait a bit
        time.sleep(0.5)

# Semaphore to limit concurrent requests
request_semaphore = threading.Semaphore(MAX_WORKERS)

def get_notebook_content_from_extracted_files(notebook_name, nb_contents_dir):
    notebook_dir = nb_contents_dir / notebook_name
    
    if not notebook_dir.exists():
        return None, None
    
    code_file = notebook_dir / 'code.txt'
    narrative_file = notebook_dir / 'narrative_md.txt'
    
    code_content = None
    narrative_content = None
    
    if code_file.exists():
        try:
            with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                code_content_raw = f.read()
            code_content = preprocess_text(code_content_raw, 'code')
        except Exception as e:
            print(f"Error reading code file for {notebook_name}: {e}")
    
    if narrative_file.exists():
        try:
            with open(narrative_file, 'r', encoding='utf-8', errors='ignore') as f:
                narrative_content_raw = f.read()
            narrative_content = preprocess_text(narrative_content_raw, 'narrative')
        except Exception as e:
            print(f"Error reading narrative file for {notebook_name}: {e}")
    
    return code_content, narrative_content

def extract_top_domain(top_domains_str):
    try:
        top_domains = ast.literal_eval(top_domains_str)
        if top_domains and len(top_domains) > 0:
            return top_domains[0]
    except (ValueError, SyntaxError):
        matches = re.findall(r"'(.*?)'", top_domains_str)
        if matches:
            return matches[0]
    
    return "Unknown"

def estimate_token_count(text):
    """Roughly estimate the number of tokens in a text string."""
    
    # Assuming an average word length of 4.7 characters in English language
    # Source:https://www.researchgate.net/figure/Average-word-length-in-the-English-language-Different-colours-indicate-the-results-for_fig1_230764201
    average_word_length = 4.7
    return len(text) / average_word_length

def query(domain, code_content, narrative_content):
    with request_semaphore:
        if isinstance(code_content, list):
            code_content = "\n".join(code_content)
        code_sample = code_content[:1500] + "..." if code_content and len(code_content) > 1500 else code_content

        if isinstance(narrative_content, list):
            narrative_content = "\n".join(narrative_content)
        narrative_sample = narrative_sample = narrative_content[:1500] + "..." if narrative_content and len(narrative_content) > 1500 else narrative_content
        
        if not code_sample and not narrative_sample:
            return "Error: Notebook content unavailable"

        prompt = (
            f"Domain to evaluate: {domain}\n\n"
            "Notebook Content:\n"
        )
        
        if code_sample:
            prompt += f"CODE:\n{code_sample}\n\n"
        
        if narrative_sample:
            prompt += f"MARKDOWN:\n{narrative_sample}\n\n"
        
        prompt += (
            "Question: Based on the notebook content above, is the given domain relevant to this notebook? "
            "How specific & precise are the domains? Are they overly general (e.g. data analysis) or "
            "appropriately specific (e.g. NLP)? Answer all the questions in plain text and keep your "
            "response concise within 50 tokens."
        )

        # Estimate token count for rate limiting
        estimated_input_tokens = estimate_token_count(prompt)
        estimated_output_tokens = 100  # Max tokens in response
        total_estimated_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Wait until we have token capacity
        wait_for_token_capacity(total_estimated_tokens)
        
        # Add jitter to request timing to avoid thundering herd
        jitter = random.uniform(0, REQUEST_INTERVAL * 0.5)
        time.sleep(REQUEST_INTERVAL + jitter)
        
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            # try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system",
                        "content": "You are a helpful assistant. Your task is to understand the content of a Jupyter Notebook and analyse the domains predicted for that notebook. After doing so, you are to make a judgedment on whether the predicted domains are indeed relevant to the content and whether the domain is specific . Answer all the questions in plain text keep your response concise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.2
            )
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            track_token_usage(time.time(), total_tokens)
            
            return response.choices[0].message.content.strip()
            
            # except Exception as e:
            #     if retry < max_retries - 1:
            #         # Use exponential backoff with jitter
            #         retry_delay = (2 ** retry) + random.uniform(0, 1)
            #         print(f"Error calling OpenAI API: {e}. Retrying in {retry_delay:.2f}s... (Attempt {retry+1}/{max_retries})")
            #         time.sleep(retry_delay)
            #     else:
            #         return f"Error: {str(e)}"

def process_notebook(row, nb_contents_dir):
    notebook_name = row['notebook']
    top_domain = extract_top_domain(row['top_domains'])
    
    code_content, narrative_content = get_notebook_content_from_extracted_files(notebook_name, nb_contents_dir)
    
    if not code_content and not narrative_content:
        return row.name, "Error: No content available"
    
    llm_response = query(top_domain, code_content, narrative_content)
    
    return row.name, llm_response

def main():
    current_dir = Path.cwd()
    csv_path = current_dir / 'domain_classification_results.csv'
    output_csv_path = current_dir / 'qualitative.csv'
    nb_contents_dir = current_dir / 'nb_contents'

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")

    if not nb_contents_dir.exists():
        raise FileNotFoundError(f"nb_contents directory not found: {nb_contents_dir}")

    df = pd.read_csv(csv_path)
    df['top_domain'] = df['top_domains'].apply(extract_top_domain)
    df['llm_relevance'] = None

    # Process in smaller chunks to better manage rate limits
    chunk_size = 20
    total_rows = len(df)
    
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk = df.iloc[i:end_idx]
        
        print(f"\nProcessing chunk {i//chunk_size + 1}/{(total_rows-1)//chunk_size + 1} (rows {i+1}-{end_idx})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(process_notebook, row, nb_contents_dir): row.name 
                for _, row in chunk.iterrows()
            }
            
            with tqdm(total=len(future_to_index), desc="Processing notebooks") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    index, llm_response = future.result()
                    df.at[index, 'llm_relevance'] = llm_response
                    pbar.update(1)
        
        # Save intermediate results after each chunk to avoid loosing data halfway in case of crash and better debugging
        df.to_csv(output_csv_path, index=False)
        print(f"Saved intermediate results to {output_csv_path}")
        
        # Add a pause between chunks to respect rate limits
        if end_idx < total_rows:
            pause_time = 5
            print(f"Pausing for {pause_time} seconds before next chunk...")
            time.sleep(pause_time)

    print(f"\nProcessing complete! Successfully created {output_csv_path} with {len(df)} rows")

if __name__ == "__main__":
    main()