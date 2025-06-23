"""
Migration script to upgrade from the current VGG16-based system to the improved CLIP-based system
"""

import os
import sqlite3
import numpy as np
from PIL import Image
from improved_pursuit import ImprovedFursuitIdentifier
from download import recall_furtrack_data_by_embedding_id
from random import sample

old_src_path = "/Users/golis001/Desktop/rutorch"
old_db_path = f"{old_src_path}/furtrack.db"
furtrack_images = f"{old_src_path}/furtrack_images"

def test_migration():
    """Test the migrated system with a sample image"""
    identifier = ImprovedFursuitIdentifier()
    
    # Find a test image
    image_dir = furtrack_images
    if os.path.exists(image_dir):
        test_files = sample([f for f in os.listdir(image_dir) if f.endswith('.jpg')], 5)
        
        for test_file in test_files:
            print(f"\nTesting with {test_file}:")
            try:
                test_image = Image.open(os.path.join(image_dir, test_file))
                results = identifier.identify_character(test_image, top_k=3)
                
                print("Results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['character_name']} "
                          f"(confidence: {result['confidence']:.3f})")
            except Exception as e:
                print(f"Error testing {test_file}: {e}")

def compare_systems():
    """Compare results between old and new systems"""
    from pursuit import detect_characters
    
    identifier = ImprovedFursuitIdentifier()
    
    # Find test images
    image_dir = furtrack_images
    if os.path.exists(image_dir):
        test_files =  sample([f for f in os.listdir(image_dir) if f.endswith('.jpg')], 3)
        
        for test_file in test_files:
            print(f"\n=== Comparing results for {test_file} ===")
            image_path = os.path.join(image_dir, test_file)
            
            try:
                # New system results
                print("New system (CLIP):")
                test_image = Image.open(image_path)
                new_results = identifier.identify_character(test_image, top_k=3)
                for i, result in enumerate(new_results, 1):
                    print(f"  {i}. {result['character_name']} "
                          f"(confidence: {result['confidence']:.3f})")
                    
                # Old system results
                print("Old system (VGG16):")
                old_results = detect_characters(image_path, 3)
                for i, result in enumerate(old_results, 1):
                    print(f"  {i}. {result['char']}")
            except Exception as e:
                print(f"Error comparing {test_file}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "migrate":
            migrate_existing_data()
        elif command == "test":
            test_migration()
        elif command == "compare":
            compare_systems()
        else:
            print("Usage: python migrate_to_improved.py [migrate|test|compare]")
    else:
        print("Available commands:")
        print("  migrate  - Migrate data from old system to new system")
        print("  test     - Test the new system with sample images")
        print("  compare  - Compare old vs new system results")
