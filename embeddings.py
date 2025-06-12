# BERT, Word2Vec, GloVe
import os
# Set environment variables to disable TensorFlow/JAX but allow downloads - This is an issue for Mac Silicon
os.environ["USE_TORCH"] = "1"  # Force PyTorch for BERT installation
os.environ["NO_TF"] = "1"  # Disable TensorFlow completely
os.environ["USE_TF"] = "0"  # Explicitly disable TF

import json
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Import for Gensim models
import gensim.downloader as api
from gensim.models import KeyedVectors

#GloVe and Word2Vec model storage
from transformers.utils import TRANSFORMERS_CACHE
print(f"Transformers models are stored in: {TRANSFORMERS_CACHE}")
import gensim
print(f"Gensim data directory: {gensim.downloader.base_dir}")

import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel


def extract_terms_from_file(file_path):
    """
    Extract terms from salient terms text files.
    Returns a dictionary mapping term types to lists of terms.
    """
    term_patterns = {
        'lda': r'Top 30 (?:Overall )?LDA Salient Terms.*?-{60}\n(.*?)(?:\n\n|\Z)',
        'bow': r'Top 30 (?:Overall )?BOW Terms.*?-{60}\n(.*?)(?:\n\n|\Z)',
        'tfidf': r'Top 30 (?:Overall )?TF-IDF Salient Terms.*?-{60}\n(.*?)(?:\n\n|\Z)'
    }
    
    term_dict = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for term_type, pattern in term_patterns.items():
            matches = re.search(pattern, content, re.DOTALL)
            if matches:
                terms_section = matches.group(1).strip()
                if "N/A" not in terms_section:
                    # Extract terms from lines like "- term: score"
                    term_lines = re.findall(r'- (.*?):', terms_section)
                    term_dict[term_type] = term_lines
    except Exception as e:
        print(f"Error extracting terms from {file_path}: {e}")
    
    return term_dict


class EmbeddingGenerator:
    """Base class for embedding generation"""
    
    def __init__(self, model_name, embedding_dim):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.loaded = False
    
    def load_model(self):
        raise NotImplementedError("Subclasses must implement load_model")
    
    def get_embedding(self, term):
        raise NotImplementedError("Subclasses must implement get_embedding")
    
    def generate_embeddings(self, terms_list):
        if not self.loaded:
            self.load_model()
        
        embeddings = {}
        for term in terms_list:
            embeddings[term] = self.get_embedding(term)
        
        return embeddings
    
    def save_embeddings(self, embeddings, output_path):
        # Create a dictionary with term-embedding pairs
        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {}
        for term, embedding in embeddings.items():
            if isinstance(embedding, np.ndarray):
                serializable_embeddings[term] = embedding.tolist()
            else:
                # For zero vectors or other formats
                serializable_embeddings[term] = [0.0] * self.embedding_dim
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, ensure_ascii=False, indent=2)


class BertEmbedder(EmbeddingGenerator):
    def __init__(self):
        super().__init__("bert-base-uncased", 768)  # 768 dimensions for BERT base
    
    def load_model(self):
        print(f"Loading {self.model_name}...")
        print("Downloading model if not already cached (this may take some time)")
        try:
            # Download model if needed (no local_files_only flag)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name) #DOWNLOADING THE TOKENIZER
            self.model = BertModel.from_pretrained(self.model_name) #DOWNLOADING THE MODEL
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            print(f"Successfully loaded {self.model_name}")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            raise RuntimeError(f"Failed to load BERT model: {e}")
    
    def get_embedding(self, term):
        try:
            # Tokenize and get token IDs
            inputs = self.tokenizer(term, return_tensors="pt", padding=True, truncation=True)
            
            # Get BERT embeddings without gradient calculation
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the [CLS] token embedding as the sentence embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            return embedding
        except Exception as e:
            print(f"Error generating BERT embedding for '{term}': {e}")
            return np.zeros(self.embedding_dim)


class Word2VecEmbedder(EmbeddingGenerator):
    def __init__(self):
        super().__init__("word2vec-google-news-300", 300)  # 300 dimensions for Google News Word2Vec
    
    def load_model(self):
        print(f"(Down)Loading {self.model_name}...")
        self.model = api.load(self.model_name) #DOWNLOADING THE WORD2VEC MODEL
        self.loaded = True
    
    def get_embedding(self, term):
        try:
            # Split into words for multi-word terms
            words = term.lower().split()
            
            # Get embeddings for each word and average them
            word_embeddings = []
            for word in words:
                if word in self.model:
                    word_embeddings.append(self.model[word])
            
            if word_embeddings:
                # Average the word embeddings
                return np.mean(word_embeddings, axis=0)
            else:
                # Return zero vector if no words were found in the model
                return np.zeros(self.embedding_dim)
        except Exception as e:
            print(f"Error generating Word2Vec embedding for '{term}': {e}")
            return np.zeros(self.embedding_dim)


class GloveEmbedder(EmbeddingGenerator):
    def __init__(self):
        super().__init__("glove-wiki-gigaword-100", 100)  # 100 dimensions for GloVe
    
    def load_model(self):
        print(f"(Down)Loading {self.model_name}...")
        self.model = api.load(self.model_name) #DOWNLOADING THE GLOVE MODEL
        self.loaded = True
    
    def get_embedding(self, term):
        try:
            # Split into words for multi-word terms
            words = term.lower().split()
            
            # Get embeddings for each word and average them
            word_embeddings = []
            for word in words:
                if word in self.model:
                    word_embeddings.append(self.model[word])
            
            if word_embeddings:
                # Average the word embeddings
                return np.mean(word_embeddings, axis=0)
            else:
                # Return zero vector if no words were found in the model
                return np.zeros(self.embedding_dim)
        except Exception as e:
            print(f"Error generating GloVe embedding for '{term}': {e}")
            return np.zeros(self.embedding_dim)


def create_special_term_embeddings(terms_list, embedders, embedding_dirs):
    """
    Create embeddings for a specific list of terms and save them in JSON format.
    
    Args:
        terms_list: List of terms to embed
        embedders: Dictionary of embedder objects
        embedding_dirs: Dictionary of embedding directory paths
    """
    print("\nGenerating embeddings for special term list...")
    
    # Process each embedding type
    for embed_type, embedder in embedders.items():
        print(f"Generating {embed_type} embeddings for special term list...")
        
        # Generate embeddings for the terms
        embeddings_dict = embedder.generate_embeddings(terms_list)
        
        # Define output file path with .json extension
        output_path = embedding_dirs[embed_type] / f"special_terms_{embed_type}.json"
        
        # Prepare dictionary for JSON output
        json_output_dict = {}
        for term, embedding in embeddings_dict.items():
            if isinstance(embedding, np.ndarray):
                # Convert numpy array to a list of floats for JSON serialization
                json_output_dict[term] = embedding.tolist()
            else:
                # Handle cases where embedding might not be a numpy array as before
                # (e.g., if generate_embeddings can return non-array types or None)
                # Assuming a default zero vector if embedding is not as expected
                json_output_dict[term] = [0.0] * getattr(embedder, 'embedding_dim', 100) # Default dim if not found
        
        # Save embeddings in JSON format
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output_dict, f, indent=2) 
        
        print(f"Saved special terms embeddings to {output_path}")

def create_embeddings():
    """
    Main function to create embeddings for salient terms.
    """
    current_dir = Path.cwd()
    salient_terms_dir = current_dir / 'salient_terms'
    
    # Create output directories for each embedding type
    embedding_dirs = {
        'bert': current_dir / 'bert_embeddings',
        'word2vec': current_dir / 'word2vec_embeddings',
        'glove': current_dir / 'glove_embeddings'
    }
    
    # Create directories if they don't exist
    for dir_path in embedding_dirs.values():
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize embedders
    embedders = {
        'bert': BertEmbedder(),
        'word2vec': Word2VecEmbedder(),
        'glove': GloveEmbedder()
    }
    
    # Walk through salient_terms directory structure
    for subfolder_path in sorted(salient_terms_dir.glob('*/')):
        subfolder_name = subfolder_path.name
        print(f"\nProcessing subfolder: {subfolder_name}")
        
        # Create corresponding subfolders in each embedding directory
        for embed_type, embed_dir in embedding_dirs.items():
            embed_subfolder = embed_dir / subfolder_name
            embed_subfolder.mkdir(exist_ok=True, parents=True)
        
        # Process salient term files
        for term_file in sorted(subfolder_path.glob('*.txt')):
            file_name = term_file.name
            print(f"  Processing file: {file_name}")
            
            # Extract terms from file
            terms_dict = extract_terms_from_file(term_file)
            
            # Generate embeddings for each term type and save
            for embed_type, embedder in embedders.items():
                for term_type, terms in terms_dict.items():
                    if terms:  # Only process if terms were found
                        # Generate embeddings
                        print(f"    Generating {embed_type} embeddings for {term_type} terms...")
                        embeddings = embedder.generate_embeddings(terms)
                        
                        # Create output filename: retain original name + add embedding type
                        output_name = f"{term_file.stem}_{term_type}_{embed_type}.json"
                        output_path = embedding_dirs[embed_type] / subfolder_name / output_name
                        
                        # Save embeddings
                        embedder.save_embeddings(embeddings, output_path)
                        print(f"    Saved to {output_path}")
    
    # Special terms list provided by the user
    special_terms = {'random forest', 'tqdm', 'sports', 'movies and tv shows', 'time series analysis', 'healthcare', 'plotly', 'ml ethics', 'dailychallenge', 'wandb', 'ensembling', 'celebrities', 'text mining', 'jobs and career', 'python', 'advanced', 'religion and belief systems', 'adversarial learning', 'eyes and vision', 'image', 'classification', 'gradient boosting', 'tensorflow', 'data analytics', 'weather and climate', 'travel', 'optimization', 'insurance', 'convolution', 'transportation', 'image augmentation', 'spaCy', 'image generator', 'education', 'randomForest', 'regression', 'india', 'selective kernel', 'bayesian statistics', 'physics', 'seaborn', 'english', 'gpu', 'real estate', 'video', 'arts and entertainment', 'exercise', 'energy', 'weight standardization', 'logistic regression', 'reinforcement learning', 'tabular', 'retail and shopping', 'multiclass classification', 'cv2', 'finnish', 'diseases', 'sklearn', 'dnn', 'nltk', 'games', 'social science', 'video games', 'multimodal', 'manufacturing', 'neural networks', 'finance', 'data visualization', 'image segmentation', 'resnet v2 50', 'gan', 'xgboost', 'categorical', 'pytorch', 'law', 'keras', 'computer vision', 'transformers', 'rnn', 'tpu', 'programming', 'atmospheric science', 'text', 'currencies and foreign exchange', 'scipy', 'business', 'matplotlib', 'comics and animation', 'utility script', 'medicine', 'software', 'cnn', 'cities and urban areas', 'artificial intelligence', 'environment', 'text pre-processing', 'audio', 'covid19', 'binary classification', 'food', 'multilabel classification', 'feature engineering', 'linear regression', 'model comparison', 'inception resnet v2', 'art', 'sampling', 'nlp', 'recommender systems', 'biology', 'text generation', 'crime', 'agriculture', 'health', 'news', 'lightgbm', 'psychology', 'diabetes', 'neuroscience', 'data cleaning', 'heart conditions', 'automl', 'animals', 'PIL', 'pandas', 'deep learning', 'torchvision', 'tidyverse', 'earth science', 'baseball', 'IPython', 'exploratory data analysis', 'beginner', 'transfer learning', 'model explainability', 'cancer', 'numpy', 'music', 'learn', 'image classification', 'lstm', 'online communities', 'intermediate', 'decision tree'}
    
    # Generate embeddings for the special terms list
    create_special_term_embeddings(special_terms, embedders, embedding_dirs)
    
    print("\nEmbedding generation complete!")


if __name__ == "__main__":
    create_embeddings()