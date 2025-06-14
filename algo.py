import os
import re
import csv
import pandas as pd
import ast
from pathlib import Path
from collections import defaultdict
import sys
# import cyclomatic

def preprocess_text(text, file_type):
    """
    Preprocesses the input text by applying file-specific cleaning rules.

    Args:
        text (str): The input text to be processed.
        file_type (str): The type of file ('code', 'comment', 'narrative', 'unknown').

    Returns:
        str: Cleaned text.
    """
    cleaned_text = ""
    lines = text.splitlines()
    cleaned_lines = []

    if file_type == 'code':
        for line in lines:
            cleaned_line = re.sub(r'^Content of code cell #\d+:', '', line).strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        cleaned_text = '\n'.join(cleaned_lines)
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
        cleaned_text = '\n'.join(cleaned_lines)
    elif file_type == 'narrative':
        skip_headers = True
        for line in lines:
            if skip_headers and ('Markdown Narrative from:' in line or '=================' in line):
                continue
            skip_headers = False
            if not line.strip() or '------------------------------' in line or line.startswith('Markdown Cell #'):
                continue
            cleaned_lines.append(line.strip())
        cleaned_text = '\n'.join(cleaned_lines)
    else: # 'unknown' or other types
        cleaned_text = text # Use original text if no specific rule

    return cleaned_text

def extract_all_domains_for_all_combined(csv_path):
    """
    Extracts domain information for all 9 variations per notebook from domain_classification_results.csv,
    specifically from rows where part='all combined'
    
    Args:
        csv_path (Path): Path to the domain classification results CSV
        
    Returns:
        dict: A dictionary mapping notebook names with variation indices to their top-1 domain
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading domain classification results from {csv_path}...")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter rows where part = 'all combined'
    all_combined_df = df[df['part'] == 'all combined']
    
    if all_combined_df.empty:
        print("No rows found with part='all combined'")
        # Fallback to using all rows if no 'all combined' part exists
        all_combined_df = df  
    
    # Dictionary to store notebook+variation -> top domain mapping
    variations_top_domains = {}
    
    # Track notebooks and their variations
    notebook_variations = defaultdict(int)
    
    # Process each row in the filtered DataFrame
    for _, row in all_combined_df.iterrows():
        notebook = row['notebook']
        
        # Create a variation index for this notebook
        variation_idx = notebook_variations[notebook]
        notebook_variations[notebook] += 1
        
        # Create a unique identifier for this notebook+variation
        notebook_with_var = f"{notebook}_var{variation_idx}"
        
        # Extract the top domain
        try:
            top_domains_str = row['top_domains']
            if isinstance(top_domains_str, str):
                try:
                    # Parse the string representation of the list
                    top_domains = ast.literal_eval(top_domains_str)
                    if top_domains and len(top_domains) > 0:
                        # Get the first domain (top-1)
                        top1_domain = top_domains[0]
                        
                        # Store in dictionary
                        variations_top_domains[notebook_with_var] = {
                            'top1_domain': top1_domain,
                            'term_type': row.get('term_type', 'unknown'),
                            'embedding_type': row.get('embedding_type', 'unknown'),
                            'base_notebook': notebook,
                            'variation_idx': variation_idx
                        }
                        print(f"Extracted top domain for {notebook_with_var}: {top1_domain}")
                except (ValueError, SyntaxError):
                    # If parsing fails, try regex approach
                    matches = re.findall(r"'(.*?)'", top_domains_str)
                    if matches:
                        top1_domain = matches[0]
                        variations_top_domains[notebook_with_var] = {
                            'top1_domain': top1_domain,
                            'term_type': row.get('term_type', 'unknown'),
                            'embedding_type': row.get('embedding_type', 'unknown'),
                            'base_notebook': notebook,
                            'variation_idx': variation_idx
                        }
                        print(f"Extracted top domain for {notebook_with_var} (regex): {top1_domain}")
        except Exception as e:
            print(f"Error parsing top_domains for {notebook_with_var}: {e}")
    
    # Check for notebooks with less than 9 variations
    for notebook, var_count in notebook_variations.items():
        if var_count < 9:
            print(f"Warning: Notebook {notebook} has only {var_count} variations (expected 9)")
            # Pad with additional variations if needed
            for i in range(var_count, 9):
                notebook_with_var = f"{notebook}_var{i}"
                # Use the domain from the first variation if available
                first_var_key = f"{notebook}_var0"
                if first_var_key in variations_top_domains:
                    first_var_info = variations_top_domains[first_var_key]
                    variations_top_domains[notebook_with_var] = {
                        'top1_domain': first_var_info['top1_domain'],
                        'term_type': f"padded_{i}",
                        'embedding_type': f"padded_{i}",
                        'base_notebook': notebook,
                        'variation_idx': i
                    }
                    print(f"Added padding variation {notebook_with_var} with domain from first variation")
                else:
                    # No first variation available, use unknown
                    variations_top_domains[notebook_with_var] = {
                        'top1_domain': 'unknown',
                        'term_type': f"padded_{i}",
                        'embedding_type': f"padded_{i}",
                        'base_notebook': notebook,
                        'variation_idx': i
                    }
                    print(f"Added padding variation {notebook_with_var} with unknown domain")
    
    print(f"Total variations extracted: {len(variations_top_domains)}")
    return variations_top_domains

def count_lines(code_content):
    """Count the number of non-empty lines in code"""
    lines = [line.strip() for line in code_content.splitlines()]
    return len([line for line in lines if line])

def extract_library_usage(code_content):
    """Extract library imports and create one-hot encoding"""
    # Common libraries to detect
    libraries = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow',
        'keras', 'torch', 'pytorch', 'scipy', 'nltk', 'cv2', 'xgboost',
        'lightgbm', 'transformers', 'huggingface', 'spacy', 'plotly',
        'gensim', 'statsmodels', 'tqdm'
    ]
    
    # Tensor libraries to specifically check (will be one-hot encoded separately)
    tensor_libraries = [
        'numpy', 'tensorflow', 'torch', 'jax', 'mxnet', 'theano', 'cupy'
    ]
    
    # Dictionary to store one-hot encoding
    library_usage = {f"lib_{lib}": 0 for lib in libraries}
    tensor_lib_usage = {f"tensor_lib_{lib}": 0 for lib in tensor_libraries}
    
    # Regular expressions for different import patterns
    import_patterns = [
        r'import\s+(\w+)',                   # import numpy
        r'from\s+(\w+)(?:\.\w+)?\s+import',  # from numpy import ...
        r'import\s+(\w+)\s+as\s+\w+',        # import numpy as np
        r'from\s+(\w+)\s+import\s+\w+',      # from numpy import array
    ]
    
    # Find all matches for all patterns
    matches = []
    for pattern in import_patterns:
        matches.extend(re.findall(pattern, code_content))
    
    # Convert matches to lowercase for case-insensitive comparison
    matches = [match.lower() for match in matches]
    
    # Mark libraries as used (1) if found in matches
    for lib in libraries:
        if lib.lower() in matches:
            library_usage[f"lib_{lib}"] = 1
        # Special case for PyTorch (could be imported as torch)
        elif lib.lower() == 'pytorch' and 'torch' in matches:
            library_usage[f"lib_{lib}"] = 1
    
    # Mark tensor libraries as used
    for lib in tensor_libraries:
        if lib.lower() in matches:
            tensor_lib_usage[f"tensor_lib_{lib}"] = 1
        # Special case for PyTorch
        elif lib.lower() == 'torch' and 'pytorch' in matches:
            tensor_lib_usage[f"tensor_lib_{lib}"] = 1
    
    # Combine both dictionaries
    return {**library_usage, **tensor_lib_usage}

# AST (Abstract Syntax Tree) parsing is preferred over regex for code analysis
# because it provides a deeper, semantic understanding of the code's structure,
# leading to more accurate identification of elements like function calls.
# AST is robust to code style variations and better handles context, reducing
# false positives/negatives common with text-based regex matching. Regex is
# used here as a fallback for code that may not be strictly parsable by AST
# (e.g., containing IPython magics or syntax errors).

def extract_tensor_operations(code_content):
    """Extract tensor operations and create one-hot encoding using AST, with regex fallback."""
    tensor_ops = [
        'dot', 'matmul', 'multiply', 'add', 'subtract', 'divide',
        'transpose', 'reshape', 'concatenate', 'stack', 'split',
        'slice', 'gather', 'scatter', 'einsum', 'conv', 'pool'
    ]
    
    tensor_op_usage = {f"op_{op}": 0 for op in tensor_ops}
    num_tensor_ops = 0

    try: # taken from https://docs.python.org/3/library/ast.html
        # Pre-filter lines that are obviously not standard Python (e.g., IPython magics)
        lines = code_content.splitlines()
        python_lines = [line for line in lines if not (line.strip().startswith('%') or line.strip().startswith('!'))]
        parsable_code_content = "\n".join(python_lines)

        if not parsable_code_content.strip():
            # If all lines were magics, there were some errors or empty, no AST parsing needed
            pass
        else:
            tree = ast.parse(parsable_code_content)
            
            class TensorOpVisitor(ast.NodeVisitor):
                def __init__(self, ops_list):
                    self.tensor_ops_set = set(ops_list) 
                    self.op_counts = {op: 0 for op in ops_list}
                    self.total_op_count = 0

                def visit_Call(self, node): #extract tensor operations as keywords!!!!!
                    func_name = None
                    if isinstance(node.func, ast.Name): # e.g., add(...)
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute): # e.g., np.add(...) or tensor.add(...)
                        func_name = node.func.attr
                    
                    if func_name and func_name.lower() in self.tensor_ops_set:
                        op_key = func_name.lower() # tensor_ops are already lowercase
                        self.op_counts[op_key] += 1
                        self.total_op_count += 1
                    
                    self.generic_visit(node)

            visitor = TensorOpVisitor(tensor_ops)
            visitor.visit(tree)
            
            num_tensor_ops = visitor.total_op_count
            for op_name, count in visitor.op_counts.items():
                if count > 0:
                    tensor_op_usage[f"op_{op_name}"] = 1
                
    except SyntaxError:
        # Fallback to regex method if AST parsing fails (even after cleaning)
        # This indicates more complex syntax issues or that the code is not valid Python.
        # The original code_content (not parsable_code_content) should be used for regex,
        # as regex might still find patterns in lines that AST couldn't parse.
        print(f"SyntaxError during AST parsing for tensor operations. Content (or part of it) might not be pure Python. Falling back to regex.")
        
        total_ops_regex = 0
        for op in tensor_ops:
            patterns = [
                rf'\b{op}\(', 
                rf'\.{op}\(',
                rf'\b{op}_'
            ]
            
            op_count_regex = 0
            for pattern in patterns:
                matches = re.findall(pattern, code_content, re.IGNORECASE) # Use original code_content
                op_count_regex += len(matches)
                
            if op_count_regex > 0:
                tensor_op_usage[f"op_{op}"] = 1
            total_ops_regex += op_count_regex
        num_tensor_ops = total_ops_regex

    tensor_op_usage['num_tensor_ops'] = num_tensor_ops
    return tensor_op_usage

def count_functions_and_classes(code_content):
    """Count number of function and class definitions"""
    function_count = len(re.findall(r'def\s+\w+\s*\(', code_content))
    class_count = len(re.findall(r'class\s+\w+\s*[(:)]', code_content))
    
    return {'num_functions': function_count, 'num_classes': class_count}

def estimate_inheritance_depth(code_content):
    """Estimate the maximum depth of inheritance"""
    # Find class definitions with inheritance
    class_defs = re.findall(r'class\s+(\w+)\s*\((.*?)\):', code_content)
    
    # If no classes found, return 0
    if not class_defs:
        return 0
    
    # Create a dictionary mapping class name to their direct parent classes
    class_inheritance = {}
    for class_name, parents_str in class_defs:
        # Split parent classes string and remove whitespace
        parents = [p.strip() for p in parents_str.split(',')]
        class_inheritance[class_name] = parents
    
    # Function to calculate depth recursively
    def get_depth(class_name, visited=None):
        if visited is None:
            visited = set()
        
        # Avoid cycles
        if class_name in visited:
            return 0
        visited.add(class_name)
        
        # If class not in our definitions, it's likely a built-in or imported class
        if class_name not in class_inheritance:
            return 1
        
        # Get parents
        parents = class_inheritance[class_name]
        if not parents:
            return 1
        
        # Find maximum depth among parents
        max_parent_depth = 0
        for parent in parents:
            parent_depth = get_depth(parent, visited.copy())
            max_parent_depth = max(max_parent_depth, parent_depth)
        
        return 1 + max_parent_depth
    
    # Calculate maximum depth across all classes
    max_depth = 0
    for class_name in class_inheritance:
        depth = get_depth(class_name)
        max_depth = max(max_depth, depth)
    
    return max_depth

def calculate_cyclomatic_complexity(code_content):
    """
    Calculate cyclomatic complexity using the formula E - N + 2P
    where:
    - E = number of edges in the control flow graph
    - N = number of nodes in the control flow graph
    - P = number of connected components (usually 1 for a single function)
    
    Simplified as: number of decision points + 1
    """# Count decision points (predicates)
    decision_points = len(re.findall(r'\bif\b|\bfor\b|\bwhile\b|\band\b|\bor\b|\belif\b|\bcatch\b|\bexcept\b|\bcase\b|\bwith\b', code_content))
    
    # Count functions and classes (representing separate components)
    function_defs = re.findall(r'def\s+\w+\s*\(', code_content)
    class_defs = re.findall(r'class\s+\w+\s*[(:)]', code_content)
    
    # Number of components (P) - at minimum 1, plus any additional disconnected functions/classes
    components = max(1, len(function_defs) + len(class_defs))
    
    # Estimate number of nodes (N) - roughly 1 per line plus decision points
    lines = [line for line in code_content.splitlines() if line.strip()]
    nodes = len(lines)
    
    # Estimate number of edges (E) - roughly nodes + decision points (each decision adds an extra edge)
    edges = nodes + decision_points
    
    # Apply the formula: E - N + 2P
    complexity = edges - nodes + (2 * components)
    
    # Ensure complexity is at least 1
    return max(1, complexity)

def process_notebook_contents():
    """Process all notebook contents in the nb_contents directory"""
    current_dir = Path.cwd()
    nb_contents_dir = current_dir / 'nb_contents'
    
    if not nb_contents_dir.exists():
        raise FileNotFoundError(f"nb_contents directory not found at {nb_contents_dir}")
    
    # Dictionary to store metrics for each notebook
    notebook_metrics = {}
    
    # Process each notebook folder
    for notebook_dir in nb_contents_dir.glob('*/'):
        notebook_name = notebook_dir.name
        print(f"Processing notebook: {notebook_name}")
        
        # Initialize metrics for this notebook
        metrics = {}
        
        # Process code file if it exists
        code_file = notebook_dir / 'code.txt'
        if code_file.exists():
            with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                code_content_raw = f.read()
            
            # Preprocess the code content
            code_content = preprocess_text(code_content_raw, 'code')
            
            # Extract metrics
            metrics['num_lines'] = count_lines(code_content)
            
            # Get library usage including tensor libraries
            library_usage = extract_library_usage(code_content)
            metrics.update(library_usage)
            
            # Get tensor operations
            tensor_ops = extract_tensor_operations(code_content)
            metrics.update(tensor_ops)
            
            # Get function and class counts
            func_class_counts = count_functions_and_classes(code_content)
            metrics.update(func_class_counts)
            
            # Calculate inheritance depth
            metrics['inheritance_depth'] = estimate_inheritance_depth(code_content)
            
            # Calculate cyclomatic complexity using improved formula
            metrics['cyclomatic_complexity'] = calculate_cyclomatic_complexity(code_content)
        else:
            # Default values if code file not found
            metrics['num_lines'] = 0
            
            # Default values for libraries - some libraries I saw manually
            for lib in ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tensorflow',
                        'keras', 'torch', 'pytorch', 'scipy', 'nltk', 'cv2', 'xgboost',
                        'lightgbm', 'transformers', 'huggingface', 'spacy', 'plotly',
                        'gensim', 'statsmodels', 'tqdm']:
                metrics[f"lib_{lib}"] = 0
            
            # Default values for tensor libraries  - https://www.jmlr.org/papers/volume20/18-277/18-277.pdf 
            for lib in ['numpy', 'tensorflow', 'torch', 'jax', 'mxnet', 'theano', 'cupy']:
                metrics[f"tensor_lib_{lib}"] = 0
            
            # Default values for tensor operations - https://www.tensorflow.org/tutorials/customization/basics
            for op in ['dot', 'matmul', 'multiply', 'add', 'subtract', 'divide',
                      'transpose', 'reshape', 'concatenate', 'stack', 'split',
                      'slice', 'gather', 'scatter', 'einsum', 'conv', 'pool']:
                metrics[f"op_{op}"] = 0
                
            metrics['num_tensor_ops'] = 0
            metrics['num_functions'] = 0
            metrics['num_classes'] = 0
            metrics['inheritance_depth'] = 0
            metrics['cyclomatic_complexity'] = 1
        
        # Process narrative markdown file if it exists
        narrative_file = notebook_dir / 'narrative_md.txt'
        if narrative_file.exists():
            with open(narrative_file, 'r', encoding='utf-8', errors='ignore') as f:
                narrative_content_raw = f.read()
            
            # Preprocess the narrative content
            narrative_content = preprocess_text(narrative_content_raw, 'narrative')
            
            # Count non-empty lines
            narrative_lines = [line for line in narrative_content.splitlines() if line.strip()]
            metrics['narrative_size'] = len(narrative_lines)
        else:
            metrics['narrative_size'] = 0
        
        # Store metrics for this notebook
        notebook_metrics[notebook_name] = metrics
    
    return notebook_metrics

def create_output_csv_with_variations(notebook_metrics, variations_domains, output_path):
    """Create the final CSV output with all metrics and top-1 domain for all variations"""
    # Initialize rows for CSV
    rows = []
    
    # Process each variation
    for notebook_with_var, domain_info in variations_domains.items():
        # Get the base notebook name without variation
        base_notebook = domain_info['base_notebook']
        
        # Check if we have metrics for this notebook
        if base_notebook in notebook_metrics:
            # Get metrics for the base notebook
            metrics = notebook_metrics[base_notebook]
            
            # Create a row with notebook name + variation
            row = {'notebook': notebook_with_var}
            
            # Add all metrics
            row.update(metrics)
            
            # Add variation information
            row['term_type'] = domain_info['term_type']
            row['embedding_type'] = domain_info['embedding_type']
            row['variation_idx'] = domain_info['variation_idx']
            row['base_notebook'] = base_notebook
            
            # Add top-1 domain
            row['top1_domain'] = domain_info['top1_domain']
            
            # Add to rows
            rows.append(row)
        else:
            print(f"Warning: No metrics found for base notebook {base_notebook} (variation {notebook_with_var})")
    
    if not rows:
        print("Error: No data to write to CSV")
        return
    
    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all field names (columns)
        fieldnames = list(rows[0].keys())
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Output CSV created at {output_path} with {len(rows)} rows")

def main():
    current_dir = Path.cwd()
    
    # Path to domain classification results CSV - this is the source for domain data
    domain_csv_path = current_dir / 'domain_classification_results.csv'
    
    # Extract all domains variations, specifically from 'all combined' part rows
    print("Extracting domains for all variations from classification results (part='all combined')...")
    variations_domains = extract_all_domains_for_all_combined(domain_csv_path)
    
    # Process notebook contents
    print("Processing notebook contents...")
    notebook_metrics = process_notebook_contents()
    
    # Create output CSV - different path from the input
    output_path = current_dir / 'code_complexity.csv'
    print("Creating output CSV with all variations...")
    create_output_csv_with_variations(notebook_metrics, variations_domains, output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()