from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def inspect_pdf():
    pdf_path = Path("scraper/data/doc_85_lZLvH0b~w7cIYI2NIUnS+LEgXt2P6wzDdai76cQSUFw=.pdf")
    
    if not pdf_path.exists():
        print("File not found, searching dir...")
        files = list(Path("scraper/data").glob("doc_85*.pdf"))
        if files:
            pdf_path = files[0]
        else:
            print("No doc_85 found.")
            return

    print(f"Inspecting: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    if not pages:
        print("No pages extracted.")
        return

    print("\n--- Page 1 Content ---")
    print(pages[0].page_content[:1000])
    
    if "רבין" in pages[0].page_content:
        print("\nSUCCESS: 'Rabin' found in text.")
    else:
        print("\nFAILURE: 'Rabin' NOT found in text.")

if __name__ == "__main__":
    inspect_pdf()
