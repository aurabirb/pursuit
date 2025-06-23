import os

# This is critical to prevent faiss from crashing on MPS
import open_clip.convert
import open_clip.utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Make it work on MacOS
os.environ['FAISS_OPT_LEVEL'] = '' # prevent it from crashing
import sys
import numpy as np
import faiss
import torch
import sqlite3
import json
from typing import cast, List, Dict, Tuple, Optional
from PIL import Image
import requests
from io import BytesIO
# from transformers import CLIPProcessor, CLIPModel
import open_clip

nfc25_db = "/Users/golis001/Desktop/nfc25-fursuits"
db_path = f"/Users/golis001/Desktop/furtrack_improved.db"
index_path = f"/Users/golis001/Desktop/faiss_improved.index"

old_src_path = "/Users/golis001/Desktop/rutorch"
old_db_path = f"{old_src_path}/furtrack.db"
furtrack_images = f"{old_src_path}/furtrack_images"
old_json_path = f"{old_src_path}/furtrack_data.json"

DEVICE_NAME = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def delete_databases():
    """Reset the new improved system by clearing the database and index"""
    print("Resetting new improved system...")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted database at {db_path}")
    else:
        print(f"No database found at {db_path}, nothing to delete.")
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"Deleted index at {index_path}")
    else:
        print(f"No index found at {index_path}, nothing to delete.")

def migrate_existing_data():
    """Migrate data from old system to new improved system"""
    
    print("Starting migration from old system to improved system...")
    
    # Initialize new system
    new_identifier = ImprovedFursuitIdentifier()
    print("Identifier initialized")
    # Load old database
    if not os.path.exists(old_db_path):
        raise FileNotFoundError(f"Old database not found at {old_db_path}. Please check the path.")
    old_conn = sqlite3.connect(old_db_path)
    old_cursor = old_conn.cursor()
    
    # Get all records with embeddings
    old_cursor.execute("""
        SELECT post_id, char, url, raw, embedding_id, date_modified
        FROM furtrack 
        WHERE embedding_id IS NOT NULL AND char != ''
        ORDER BY char
    """)
    
    old_records = old_cursor.fetchall()
    old_conn.close()
    
    print(f"Found {len(old_records)} records to migrate")
    # example_old_record = old_records[0]
    # example_old_record_raw = json.loads(example_old_record[3])
    
    # Group by character for batch processing
    character_groups = {}
    skipped_raw_count = 0
    skipped_missing_file_count = 0
    for (post_id, char, url, raw, embedding_id, date_modified) in old_records:
        if json.loads(raw).get('success') != True:
            skipped_raw_count += 1
            continue
        if char not in character_groups:
            character_groups[char] = []
        # Check if the image file exists
        img_exists = os.path.exists(f"{furtrack_images}/{post_id}.jpg")
        if not img_exists:
            skipped_missing_file_count += 1
            continue
        character_groups[char].append({
            'post_id': post_id,
            'url': url,
            'embedding_id': embedding_id,
            'date_modified': date_modified,
            'raw': json.loads(raw)  # Store raw data as dict
        })
    
    print(f"SQLITE: Found {len(character_groups)} unique characters, skipped {skipped_raw_count} records")
    print(f"SQLITE: Missing {skipped_missing_file_count} images")

    old_json_data = cast(dict, json.loads(open(old_json_path, 'r').read()))
    print(f"Loaded {len(old_json_data.keys())} records from the JSON dump")
    old_json_groups = {}
    old_json_skipped_count = 0
    old_json_missing_file_count = 0
    for post_id, jd in old_json_data.items():
        if jd.get('raw', {}).get('success') != True:
            old_json_skipped_count += 1
            continue
        # char_tags = get_char_tags(jd.get('tags', {}))
        # if not check_single_character_tags(jd.get('tags', {})):
        #     old_json_skipped_count += 1
        #     continue
        # char = char_tags[0]
        char = jd.get('char')
        if not char:
            old_json_skipped_count += 1
            continue
        img_exists = os.path.exists(f"{furtrack_images}/{post_id}.jpg")
        if not img_exists:
            old_json_missing_file_count += 1
            continue
        if char not in old_json_groups:
            old_json_groups[char] = []
        old_json_groups[char].append({
            'post_id': post_id,
            'url': jd.get('url'),
            'raw': jd.get('raw')
        })
        
    from pprint import pprint
    print(f"JSONDATA: Found {len(old_json_groups)} characters, skipped {old_json_skipped_count}")
    print(f"JSONDATA: Missing {old_json_missing_file_count} images")
    pprint(old_json_data["3004"])

    # Merge old JSON data into character groups
    for char, records in old_json_groups.items():
        if char not in character_groups:
            character_groups[char] = []
        for record in records:
            # Check if image file exists locally
            local_path = f"{furtrack_images}/{record['post_id']}.jpg"
            if os.path.exists(local_path):
                character_groups[char].append({
                    'post_id': record['post_id'],
                    'url': record['url'],
                    'raw': record['raw']
                })
            else:
                print(f"Skipping {record['post_id']} - image file does not exist")
    
    print(f"Total characters after merging JSON data: {len(character_groups)}")
    # Process each character
    migrated_count = 0
    for char_name, records in character_groups.items():
        print(f"Migrating character: {char_name} ({len(records)} images)")
        
        image_paths = []
        for record in records:
            # Check if image file exists locally
            local_path = f"{furtrack_images}/{record['post_id']}.jpg"
            if os.path.exists(local_path):
                image_paths.append(local_path)
            elif record['url']:
                # Use URL if local file doesn't exist
                image_paths.append(record['url'])
        
        if image_paths:
            added = new_identifier.add_character_images(
                character_name=char_name,
                image_paths=image_paths,
                batch_size=16
            )
            migrated_count += added
    
    print(f"Migration completed! Migrated {migrated_count} total images")
    
    # Print new system stats
    stats = new_identifier.get_character_stats()
    print("\nNew system stats:")
    print(f"Total images: {stats['total_images']}")
    print(f"Unique characters: {stats['unique_characters']}")
    print(f"Index size: {stats['index_size']}")


class ImprovedFursuitIdentifier:
    def __init__(self, 
                #  model_name: str = "openai/clip-vit-base-patch32",
                 model_name: str = "ViT-B-32",
                 db_path: str = db_path,
                 index_path: str = index_path,
                 embedding_dim: int = 512):
        
        self.model_name = model_name
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        
        # Initialize CLIP model

        print("Using device: " + DEVICE_NAME)
        self.device = torch.device(DEVICE_NAME)
        # print(f"Loading model {model_name}...")
        # self.model = cast(CLIPModel, CLIPModel.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device))
        # print(f"Model loaded: {model_name} with embedding dimension {self.embedding_dim}")
        # print("Loading processor...")
        # # requires accelerate: https://huggingface.co/docs/transformers/main/en/big_models
        # self.processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True))
        # self.model.eval()
        # print("Model evaluated.")


        # model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
        
        # pretrained also accepts local paths
        # ViT-H/14-quickgelu
        print(f"Loading model {self.model_name}...")
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained='laion2b_s34b_b79k', precision='fp32', device=self.device)
        self.model = cast(open_clip.CLIP, model)
        print(f"Loaded model with embedding dimension: {self.model.visual.output_dim}")
        print(f"expected embedding dimension: {self.embedding_dim}")
        self.processor = preprocess
        self.model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        print("Model evaluated.")
        
        # Initialize tokenizer
        print("Loading tokenizer...")
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        # self.tokenizer = cast(CLIPTokenizer, CLIPTokenizer.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True))
        

        # Initialize database and index
        print(f"Initializing database at {self.db_path}")
        self._init_database()
        print(f"Initializing FAISS index")
        self.index = self.load_or_create_index()
        
        # import torchvision.models as models
        # import torchvision.transforms as T
        # self.transform = T.Compose([
        #     T.Resize((224, 224)),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
    
    def _init_database(self):
        """Initialize SQLite database with improved schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if False:
            c.execute("""CREATE TABLE furtrack (
                      post_id text PRIMARY KEY,
                      char text,
                      url text,
                      raw text,
                      embedding_id integer default NULL,
                      date_modified timestamp DEFAULT CURRENT_TIMESTAMP
                      )""")
        c.execute("""
            CREATE TABLE IF NOT EXISTS fursuits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT UNIQUE,
                character_name TEXT,
                image_url TEXT,
                embedding_id INTEGER UNIQUE,
                confidence_score REAL DEFAULT 0.0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_character_name ON fursuits(character_name)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_id ON fursuits(embedding_id)
        """)
        
        conn.commit()
        conn.close()
    
    def load_or_create_index(self) -> faiss.Index:
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path):
            print(f"Loading existing index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            print(f"Index loaded with {index.ntotal} vectors")
        else:
            print(f"Creating new index with dimension {self.embedding_dim} at {self.index_path}")
            # Use IndexHNSWFlat for better performance on large datasets
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 50
        return index
    
    def save_index(self):
        """Save FAISS index to disk"""
        print(f"Saving index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)

    def generate_embedding_from_image(self, image: Image.Image) -> np.ndarray:
        """Extract CLIP features from image"""
        with torch.no_grad():
            # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # image_features = self.model.get_image_features(**inputs)
            image_features = self.model.encode_image(self.processor(image).unsqueeze(0).to(self.device))
            # Normalize features for better similarity search
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()
    
    @staticmethod
    def load_image_from(img_path: str) -> Image.Image:
        """Load and preprocess image for CLIP model"""
        # Load and process image
        if img_path.startswith('https://'):
            response = requests.get(img_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(img_path)
        # image = image.convert("RGB")  # Ensure RGB format
        # image = image.resize((224, 224))  # Resize to match CLIP input size
        return image

    def add_character_images(self, 
                           character_name: str, 
                           image_paths: List[str],
                           batch_size: int = 32) -> int:
        """Add multiple images for a character to the database"""
        added_count = 0
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            batch_metadata = []
            
            for img_path in batch_paths:
                try:
                    # Preprocess image
                    image = self.load_image_from(img_path)
                    if image is None:
                        print(f"Failed to load image {img_path}, skipping")
                        continue
                    # Extract features
                    embedding = self.generate_embedding_from_image(image)
                    if embedding is None or len(embedding) != self.embedding_dim:
                        print(f"Invalid embedding for {img_path}, skipping")
                        continue
                    
                    batch_embeddings.append(embedding)
                    batch_metadata.append({
                        'path': img_path,
                        'character_name': character_name,
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Add to FAISS index
            if batch_embeddings:
                embeddings_array = np.array(batch_embeddings, dtype=np.float32)
                start_id = self.index.ntotal
                self.index.add(embeddings_array) # this is sus
                
                # Add to database
                self._add_to_database(batch_metadata, start_id)
                added_count += len(batch_embeddings)
                
                print(f"Added {len(batch_embeddings)} images for {character_name}, last ID: {self.index.ntotal - 1}")
        
        self.save_index()
        return added_count

    
    def _add_to_database(self, metadata_list: List[Dict], start_embedding_id: int):
        """Add metadata to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for i, metadata in enumerate(metadata_list):
            embedding_id = start_embedding_id + i

            # TODO: add confidence score
            c.execute("""
                INSERT OR REPLACE INTO fursuits 
                (post_id, character_name, image_url, embedding_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                os.path.basename(metadata['path']),
                metadata['character_name'],
                metadata['path'],
                embedding_id,
                json.dumps(metadata)
            ))
        
        conn.commit()
        conn.close()
    
    def identify_character(self, 
                          image: Image.Image, 
                          top_k: int = 5,
                          min_confidence: float = 0.0) -> List[Dict]:
        """Identify character in image"""
        # Extract features from query image
        query_embedding = self.generate_embedding_from_image(image)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k * 2)  # Get more for filtering
        
        # Get metadata from database
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            c.execute("SELECT * FROM fursuits WHERE embedding_id = ?", (int(idx),))
            row = c.fetchone()
            
            if row:
                # Calculate confidence score (inverse of distance, normalized)
                confidence = max(0.0, 1.0 - distance / 2.0)  # Adjust normalization as needed
                
                if confidence >= min_confidence:
                    results.append({
                        'character_name': row[2],
                        'confidence': confidence,
                        'distance': float(distance),
                        'post_id': row[1],
                        'image_url': row[3]
                    })
        
        conn.close()
        
        # Sort by confidence and return top_k
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]
    
    def get_character_stats(self) -> Dict:
        """Get statistics about characters in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM fursuits")
        total_images = c.fetchone()[0]
        
        c.execute("SELECT COUNT(DISTINCT character_name) FROM fursuits")
        unique_characters = c.fetchone()[0]
        
        c.execute("""
            SELECT character_name, COUNT(*) as count 
            FROM fursuits 
            GROUP BY character_name 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_characters = c.fetchall()
        
        conn.close()
        
        return {
            'total_images': total_images,
            'unique_characters': unique_characters,
            'top_characters': top_characters,
            'index_size': self.index.ntotal
        }


# Example usage and testing
def main():
    if sys.argv[1] == "--migrate":
        delete_databases()
        migrate_existing_data()
    sys.exit(0)
    if sys.argv[1] == "--add":
        old_database_path = JSON_PATH
        db = json.loads(open(old_database_path, 'r').read())
        print(f"Loaded {len(db)} records from old database")
        # Initialize the improved system
        identifier = ImprovedFursuitIdentifier()
        # Example: Add images for a character
        identifier.add_character_images(
            character_name="fluxpaw",
            image_paths=["path/to/image1.jpg", "path/to/image2.jpg"],
        )
        sys.exit(0)

    # Initialize the improved system
    identifier = ImprovedFursuitIdentifier()
    
    # Print current stats
    stats = identifier.get_character_stats()
    print("Current database stats:")
    print(f"Total images: {stats['total_images']}")
    print(f"Unique characters: {stats['unique_characters']}")
    print(f"Index size: {stats['index_size']}")

    
    # Example: Identify character in new image
    # test_image = Image.open("test_image.jpg")
    # results = identifier.identify_character(test_image, top_k=3)
    # print("Identification results:", results)

if __name__ == "__main__":
    main()
