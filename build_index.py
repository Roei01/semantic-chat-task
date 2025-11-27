import re
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

logging.getLogger("pypdf").setLevel(logging.ERROR)

DOCS_DIR = Path("scraper/data")
VECTOR_DB_DIR = Path("vectorstore")

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()

def extract_metadata_from_filename(filename: str) -> dict:
    parts = filename.split('_', 2)
    display_name = filename
    if len(parts) >= 3:
        rest = parts[2]
        stem = Path(rest).stem
        display_name = stem.replace('_', '/')
    
    return {
        "filename": filename,
        "display_name": display_name
    }

def load_documents() -> List[Document]:
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Documents folder not found: {DOCS_DIR}")

    all_docs: List[Document] = []

    print(f"Scanning documents in {DOCS_DIR}...")
    for path in DOCS_DIR.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        loader = None
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix in (".doc", ".docx"):
                loader = Docx2txtLoader(str(path))
            else:
                continue

            print(f"Loading {path.name} ...")
            docs = loader.load()
            
            meta = extract_metadata_from_filename(path.name)
            
            for d in docs:
                d.page_content = clean_text(d.page_content)
                d.metadata["source_path"] = str(path)
                d.metadata.update(meta)

            all_docs.extend(docs)
        except Exception as e:
            msg = str(e)
            if "File is not a zip file" in msg and suffix in (".doc", ".docx"):
                continue
            print(f"Error loading {path.name}: {e}")

    print(f"Loaded {len(all_docs)} raw document pages/sections.")
    return all_docs

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} chunks from {len(documents)} source pages.")
    return split_docs

def build_vector_store(chunks: List[Document]):
    if not chunks:
        print("No chunks to index.")
        return

    VECTOR_DB_DIR.mkdir(exist_ok=True)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Building vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name="verdicts"
    )
    
    try:
        if hasattr(vectordb, 'persist'):
            vectordb.persist()
    except Exception as e:
        print(f"Persist note: {e}")
        
    print(f"Vector store successfully saved to: {VECTOR_DB_DIR}")

def main():
    documents = load_documents()
    
    if not documents:
        print("No documents found to index.")
        return

    chunks = split_documents(documents)
    build_vector_store(chunks)

if __name__ == "__main__":
    main()
