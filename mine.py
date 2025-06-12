import os
import json
import nbformat
from pathlib import Path
import re

def parse_notebook_code(notebook_path): # https://stackoverflow.com/a/78123424
    
    nb_contents_dir = Path.cwd() / 'nb_contents'
    nb_contents_dir.mkdir(exist_ok=True)
    notebook_name = Path(notebook_path).stem
    notebook_dir = nb_contents_dir / notebook_name
    notebook_dir.mkdir(exist_ok=True)

    code_file_path = notebook_dir / 'code.txt'
    
    ntbk = nbformat.read(notebook_path, nbformat.NO_CONVERT)
    nb_code_cells = [cell for cell in (ntbk.cells) if cell.cell_type == 'code']
    code_cells_input_code_up_to_limit = [cell.source for indx, cell in enumerate(nb_code_cells)]
    
    with open(code_file_path, 'w', encoding='utf-8') as f:
        for i,each_input in enumerate(code_cells_input_code_up_to_limit):
            # print(f"\nContent of code cell #{i+1}:")
            # print(each_input)
            f.write(f"\nContent of code cell #{i+1}:")
            f.write(each_input)
            
    parse_notebook_comment(code_file_path) # obtaining the in-line comments

    return ntbk

def parse_notebook_narrative_md(notebook_path):
    nb_contents_dir = Path.cwd() / 'nb_contents'
    nb_contents_dir.mkdir(exist_ok=True)
    
    notebook_name = Path(notebook_path).stem
    notebook_dir = nb_contents_dir / notebook_name
    notebook_dir.mkdir(exist_ok=True)

    md_file_path = notebook_dir / 'narrative_md.txt'
    
    ntbk = nbformat.read(notebook_path, nbformat.NO_CONVERT)
    
    md_contents = []
    for cell in ntbk.cells:
        if cell.cell_type == 'markdown':
            if hasattr(cell, 'source'):
                md_contents.append(cell.source)
            elif isinstance(cell, str):
                md_contents.append(cell)
            else:
                print(f"Warning: Unexpected cell format in {notebook_path}")
    
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Markdown Narrative from: {notebook_path}\n")
        f.write("="*50 + "\n\n")
        
        for i, md_content in enumerate(md_contents):
            f.write(f"Markdown Cell #{i+1}:\n")
            f.write(md_content)
            f.write("\n\n" + "-"*30 + "\n\n")
    
    return ntbk


def parse_notebook_comment(code_file_path):
    code_dir = code_file_path.parent
    comments_file_path = code_dir / 'comment.txt'
    
    with open(code_file_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    cell_pattern = r'Content of code cell #(\d+):(.*?)(?=Content of code cell #\d+:|$)'
    code_cells = re.findall(cell_pattern, code_content, re.DOTALL)
    
    all_comments = []
    
    for cell_num, cell_code in code_cells:
        inline_comments = re.findall(r'#\s*(.*?)$|#([^#\s].*?)$', cell_code, re.MULTILINE)
        inline_comments = [comment[0] or comment[1] for comment in inline_comments]
        
        cell_comments = [comment.strip() for comment in inline_comments if comment.strip()]
        
        if cell_comments:
            all_comments.append((int(cell_num), cell_comments))
    
    with open(comments_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Comments extracted from: {code_file_path.name}\n")
        f.write("="*50 + "\n\n")
        
        if all_comments:
            for cell_num, comments in all_comments:
                f.write(f"Comments from Code Cell #{cell_num}:\n")
                for comment in comments:
                    f.write(f"- {comment}\n")
                f.write("\n" + "-"*30 + "\n\n")
        else:
            f.write("No comments found in the code.\n")
    
    print(f"Extracted comments saved to {comments_file_path}")
    return comments_file_path

def explore_directories():
    current_dir = Path.cwd()
    data_dir = current_dir / 'data'
    
    if not data_dir.exists():
        print(f"Data directory not found at {data_dir}")
        return
    
    notebooks_found = []
    
    for folder in data_dir.iterdir():
        if folder.is_dir():
            print(f"Processing folder: {folder.name}")
            
            for subfolder in folder.iterdir():
                if subfolder.is_dir():
                    print(f"  Processing subfolder: {subfolder.name}")
                    
                    for notebook_file in subfolder.glob('*.ipynb'):
                        print(f"    Found notebook: {notebook_file.name}")
                        notebooks_found.append(notebook_file)
                        notebook_code = parse_notebook_code(notebook_file)
                        notebook_narrative_md = parse_notebook_narrative_md(notebook_file)
        
    
    print(f"\nTotal notebooks found: {len(notebooks_found)}")
    return notebooks_found

if __name__ == "__main__":
    notebooks = explore_directories()