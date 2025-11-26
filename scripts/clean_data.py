import os
import glob

def clean_data():
    docs_dir = os.path.abspath("Snippet generation/parsed_pages")
    files = glob.glob(os.path.join(docs_dir, "*.txt"))
    
    target_string = "Updated on 2025-11-22 – Authors: Kondra Vignesh, Jasmehar Singh, Malireddy Mahendra"
    
    count = 0
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if target_string in content:
                new_content = content.replace(target_string, "")
                with open(f_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            
    print(f"Cleaned {count} files.")

if __name__ == "__main__":
    clean_data()
