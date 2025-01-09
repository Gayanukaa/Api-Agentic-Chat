import os
import json
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_chunks_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Use RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)

    metadata = []
    for chunk in chunks:
        # Detect if the chunk contains an API body (e.g., JSON-like content)
        body = None
        if "{" in chunk and "}" in chunk:
            body_start = chunk.find("{")
            body_end = chunk.rfind("}") + 1
            body = chunk[body_start:body_end]

        metadata.append({
            "file_name": os.path.basename(pdf_path),
            "body": body,
        })

    return chunks, metadata

def process_reference_docs(input_folder, output_chunks_file):
    all_texts = []
    all_metadata = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_path}...")

            chunks, metadata = extract_chunks_from_pdf(file_path)
            all_texts.extend(chunks)
            all_metadata.extend(metadata)

    # Save chunks and metadata
    with open(output_chunks_file, "w", encoding="utf-8") as f:
        json.dump({"texts": all_texts, "metadata": all_metadata}, f, indent=4)
    print(f"Saved structured chunks to {output_chunks_file}")

if __name__ == "__main__":
    reference_docs_folder = "reference_docs"
    output_chunks_file = "data/chunks/chunked_data.json"
    os.makedirs(os.path.dirname(output_chunks_file), exist_ok=True)
    process_reference_docs(reference_docs_folder, output_chunks_file)
