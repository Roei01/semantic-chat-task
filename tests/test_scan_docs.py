from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from pathlib import Path
import os

def find_text_in_docs(keyword="רבין"):
    docs_dir = Path("scraper/data")
    files = list(docs_dir.glob("*.*"))
    print(f"Scanning {len(files)} files for '{keyword}'...")
    
    found_count = 0
    for f in files:
        try:
            text = ""
            if f.suffix == ".pdf":
                loader = PyPDFLoader(str(f))
                pages = loader.load()
                text = "".join([p.page_content for p in pages])
            elif f.suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(str(f))
                pages = loader.load()
                text = "".join([p.page_content for p in pages])
            
            if keyword in text:
                print(f"\nFOUND '{keyword}' in: {f.name}")
                idx = text.find(keyword)
                start = max(0, idx - 50)
                end = min(len(text), idx + 50)
                print(f"Snippet: ...{text[start:end]}...")
                found_count += 1
        except Exception as e:
            pass

    if found_count == 0:
        print(f"\nKeyword '{keyword}' NOT found in any downloaded document.")
    else:
        print(f"\nFound keyword in {found_count} documents.")

if __name__ == "__main__":
    find_text_in_docs()
