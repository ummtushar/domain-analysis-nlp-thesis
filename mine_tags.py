import os
import json
from tqdm import tqdm  # For progress tracking

def extract_kaggle_tags(data_dir="data"):
    """
    Extracts "keywords" from all kernel-metadata.json files in the nested folder structure:
    data/username/notebook-name/kernel-metadata.json
    
    Args:
        data_dir (str): Path to the data directory (default is "data" in current directory)
        
    Returns:
        list: A list of all Kaggle tags extracted from the notebooks
    """
    all_tags = []
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist")
        return all_tags
    
    # Get all user folders (first level subdirectories)
    user_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Found {len(user_folders)} user folders")
    
    # Process each user folder
    for user in tqdm(user_folders, desc="Processing user folders"):
        user_path = os.path.join(data_dir, user)
        
        # Get all notebook folders for this user
        notebook_folders = [f for f in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, f))]
        
        # Process each notebook folder
        for notebook in notebook_folders:
            notebook_path = os.path.join(user_path, notebook)
            json_path = os.path.join(notebook_path, "kernel-metadata.json")
            
            # Check if kernel-metadata.json exists
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Extract the "keywords" field if it exists
                    if "keywords" in metadata and metadata["keywords"]:
                        all_tags.extend(metadata["keywords"])
                except Exception as e:
                    print(f"Error processing {json_path}: {e}")
    
    return all_tags

def main():
    # Extract tags using default data directory
    tags_list = extract_kaggle_tags()
    
    # Print some statistics
    print(f"\nExtracted {len(tags_list)} tags in total")
    
    # Count unique tags
    unique_tags = set(tags_list)
    print(f"Found {len(unique_tags)} unique tags")

    print(unique_tags)
    
    return list(unique_tags)

if __name__ == "__main__":
    tags_list = main()