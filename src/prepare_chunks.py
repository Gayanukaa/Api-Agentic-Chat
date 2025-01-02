import os
import json
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

REFERENCE_DOCS_PATH = "reference_docs"
OUTPUT_PATH = "data/chunks/chunked_data.json"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text, chunk_size=500):
    """Split text into smaller chunks."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def process_reference_docs():
    """Process all reference PDFs and save the chunks."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    all_chunks = []

    for file_name in tqdm(os.listdir(REFERENCE_DOCS_PATH), desc="Processing PDFs"):
        file_path = os.path.join(REFERENCE_DOCS_PATH, file_name)
        if file_path.endswith(".pdf"):
            print(f"Processing {file_name}...")
            text = extract_text_from_pdf(file_path)
            chunks = split_text_into_chunks(text)
            all_chunks.extend({"file_name": file_name, "chunk": chunk} for chunk in chunks)

    # Save chunks to JSON
    with open(OUTPUT_PATH, "w") as output_file:
        json.dump(all_chunks, output_file, indent=2)
    print(f"Processed chunks saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_reference_docs()

