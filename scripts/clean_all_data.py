import os
import glob

def clean_all_files():
    """Remove author metadata from ALL text files in the project"""
    base_dir = os.path.abspath(".")
    
    target_string = "Updated on 2025-11-22 – Authors: Kondra Vignesh, Jasmehar Singh, Malireddy Mahendra"
    
    # Search in all subdirectories
    count = 0
    total_checked = 0
    
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                total_checked += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if target_string in content:
                        new_content = content.replace(target_string + '\n', '')
                        new_content = new_content.replace(target_string, '')
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        count += 1
                        print(f"Cleaned: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    print(f"\nTotal files checked: {total_checked}")
    print(f"Files cleaned: {count}")

if __name__ == "__main__":
    clean_all_files()
