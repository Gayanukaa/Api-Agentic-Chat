import os
import json
from pypdf import PdfReader

def extract_chunks_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    current_chunk = []
    current_metadata = {
        "file_name": os.path.basename(pdf_path),
    }

    for page in reader.pages:
        text = page.extract_text()
        lines = text.split("\n")

        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.split(" ")
                if line.startswith(("GET", "POST", "PUT", "DELETE")) and len(parts) > 1:  # Detect API endpoint definitions
                    if current_chunk:
                        chunks.append({
                            "text": " ".join(current_chunk),
                            "metadata": current_metadata.copy()
                        })
                        current_chunk = []

                    # Start a new chunk for the endpoint
                    current_metadata["endpoint"] = parts[0] + " " + parts[1]
                elif line.startswith("{") or line.startswith("["):  # Detect JSON body
                    current_metadata["body"] = line.strip()
                else:
                    # Avoid adding duplicate endpoint information to the description
                    if not line.startswith(("GET", "POST", "PUT", "DELETE")):
                        current_chunk.append(line.strip())

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "metadata": current_metadata
        })

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
