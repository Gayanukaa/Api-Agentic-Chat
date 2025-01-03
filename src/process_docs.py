import os
import json
from pypdf import PdfReader

def extract_chunks_from_pdf(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # Split text into chunks
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunk_metadata = {
            "file_name": os.path.basename(pdf_path),
            "chunk_start": i,
            "chunk_end": i + len(chunk_text.split()),
        }
        chunks.append({"text": chunk_text, "metadata": chunk_metadata})

    return chunks

def process_reference_docs(input_folder, output_file):
    all_chunks = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path}...")
            chunks = extract_chunks_from_pdf(file_path)
            all_chunks.extend(chunks)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
    print(f"Saved structured chunks to {output_file}")

if __name__ == "__main__":
    reference_docs_folder = "reference_docs"
    output_chunks_file = "data/chunks/chunked_data.json"
    os.makedirs(os.path.dirname(output_chunks_file), exist_ok=True)
    process_reference_docs(reference_docs_folder, output_chunks_file)
