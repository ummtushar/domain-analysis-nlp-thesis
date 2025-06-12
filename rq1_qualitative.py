# No multi-threading
import pandas as pd
import os
import re
import ast
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import salient_terms
from openai import OpenAI

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_nvidia = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

def get_notebook_content_from_extracted_files(notebook_name, nb_contents_dir):
    """
    Gets notebook content from the extracted files in the nb_contents directory.
    Uses the structure created by the mine.py script.
    
    Args:
        notebook_name (str): The name of the notebook folder.
        nb_contents_dir (Path): Path to the nb_contents directory.
        
    Returns:
        tuple: (code_content, narrative_content)
    """
    notebook_dir = nb_contents_dir / notebook_name
    
    if not notebook_dir.exists():
        print(f"Warning: Notebook directory not found: {notebook_dir}")
        return None, None
    
    code_file = notebook_dir / 'code.txt'
    narrative_file = notebook_dir / 'narrative_md.txt'
    
    code_content = None
    narrative_content = None
    
    if code_file.exists():
        try:
            with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                code_content_raw = f.read()
            code_content = salient_terms.preprocess_text(code_content_raw, 'code')
        except Exception as e:
            print(f"Error reading code file for {notebook_name}: {e}")
    
    # Read narrative content if available
    if narrative_file.exists():
        try:
            with open(narrative_file, 'r', encoding='utf-8', errors='ignore') as f:
                narrative_content_raw = f.read()
            narrative_content = salient_terms.preprocess_text(narrative_content_raw, 'narrative')
        except Exception as e:
            print(f"Error reading narrative file for {notebook_name}: {e}")
    
    return code_content, narrative_content

def extract_top_domain(top_domains_str):
    try:
        top_domains = ast.literal_eval(top_domains_str)
        if top_domains and len(top_domains) > 0:
            return top_domains[0]
    except (ValueError, SyntaxError):
        # If parsing fails, try regex approach
        matches = re.findall(r"'(.*?)'", top_domains_str)
        if matches:
            return matches[0]
    
    return "Unknown"

def query_llama_with_groq(domain, code_content, narrative_content, groq_client):
    """
    Args:
        domain (str): The domain to evaluate relevance for.
        code_content (str): The notebook's code content.
        narrative_content (str): The notebook's narrative (markdown) content.
        groq_client (Groq): The Groq client for API calls.
        
    Returns:
        str: The LLM's response about domain relevance.
    """
    if isinstance(code_content, list):
        code_content = "\n".join(code_content)
    # Prepare content for the prompt
    code_sample = code_content[:1500] + "..." if code_content and len(code_content) > 1500 else code_content

    if isinstance(narrative_content, list):
        narrative_content = "\n".join(narrative_content)
    narrative_sample = narrative_content[:1500] + "..." if narrative_content and len(narrative_content) > 1500 else narrative_content
    
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
        "Question: Based on the notebook content, is the given domain relevant to this notebook and determine whether it is highly relevant, partially relevant or just relevant- and explain why? How specific \& precise are the domains? Are they too general (e.g. data analysis) or appropriate specific (e.g. NLP) or too specific (e.g. tensor)- choose one and explain?"

    )

    # try:
    #     chat_completion = groq_client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt,
    #             }
    #         ],
    #         model="llama3-8b-8192",  # Using Llama 3 8B model
    #         max_tokens=100,  
    #         temperature=0.2  # For more consistent/factual answers
    #     )
    #     response = chat_completion.choices[0].message.content.strip()
    #     # Ensure response is not overly long
    #     return " ".join(response.split())

    # except Exception as e:
    #     print(f"Error calling Groq API")
    #     completion = client_nvidia.chat.completions.create(
    #             model="meta/llama3-8b-instruct",
    #             messages=[{"role":"user","content":prompt}],
    #             temperature=0.5,
    #             top_p=1,
    #             max_tokens=1024,
    #             stream=True
    #             )
    #     response = ""
    #     for chunk in completion:
    #         if chunk.choices[0].delta.content is not None:
    #             response += chunk.choices[0].delta.content
        
    #     return response

    model = ["llama3-8b-8192"]
    
    for retry in range(3):
            try:
                # Log which model we're trying
                print(f"Attempting with model: {model}")
                
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Your task is to understand the content of a Jupyter Notebook and analyse the domains predicted for that notebook. After doing so, you are to make a judgedment on whether the predicted domains are indeed relevant to the content and whether the domain is specific . Answer all the questions in plain text keep your response concise."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model,
                    max_tokens=100,
                    temperature=0.2
                )
                
                response = chat_completion.choices[0].message.content.strip()
                # Ensure response is not overly long
                return " ".join(response.split())
                
            except Exception as e:
                error_type = type(e).__name__
                print(f"Error with model {model}: {error_type} - {str(e)}")
                

def main():
    current_dir = Path.cwd()
    
    csv_path = current_dir / 'domain_classification_results.csv'
    
    output_csv_path = current_dir / 'qualitative_model_response.csv'
    
    nb_contents_dir = current_dir / 'nb_contents'
    
    
    df = pd.read_csv(csv_path)
    
    
    filtered_df = df.copy() 

    
    filtered_df['top_domain'] = filtered_df['top_domains'].apply(extract_top_domain)
    filtered_df['llm_relevance'] = None
    
    for index, row in filtered_df.iterrows():
        notebook_name = row['notebook']
        top_domain = row['top_domain']
        
        print(f"\nProcessing {index+1}/{len(filtered_df)}: {notebook_name} with domain '{top_domain}'")
        
        code_content, narrative_content = get_notebook_content_from_extracted_files(notebook_name, nb_contents_dir)
        
        if not code_content and not narrative_content:
            print(f"Warning: No content found for notebook {notebook_name}")
            filtered_df.at[index, 'llm_relevance'] = "Error: No content available"
            continue
        
        llm_response = query_llama_with_groq(top_domain, code_content, narrative_content, client)
        
        filtered_df.at[index, 'llm_relevance'] = llm_response
        print(f"LLM Response: {llm_response}")
    
    try:
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully created {output_csv_path} with {len(filtered_df)} rows")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main()