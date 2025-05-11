import fitz
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from transformers import AutoTokenizer

def load_tokenizer_from_huggingface(model_name: str = "meta-llama/Llama-3.1-8B") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

llama_tokenizer = load_tokenizer_from_huggingface()

def chunk_text_with_tokenizer(text: str, tokenizer, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i : i + chunk_size]
        decoded_chunk = tokenizer.decode(chunk)
        chunks.append(decoded_chunk)
    return chunks


def extract_text_from_pdf(file_path: Path, preserve_layout: bool = True) -> str:

    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        if preserve_layout:
            blocks = page.get_text("blocks", sort=True)
            text += "\n".join(block[4] for block in blocks)
        else:
            text += page.get_text()

    doc.close()
    return text


def extract_tables_from_pdf(file_path: Path) -> List[str]:

    doc = fitz.open(file_path)
    tables_text = []

    for page in doc:
        tabs = page.find_tables()
        if tabs.tables:
            for table in tabs.tables:
                # Extract table content as text
                table_data = table.extract()
                # Convert to text representation (rows separated by newlines, cells by tabs)
                table_text = "\n".join("\t".join(str(cell) for cell in row) for row in table_data)
                tables_text.append(table_text)

    doc.close()
    return tables_text


def load_all_pdfs(pdf_dir: Path, handle_tables: bool = True) -> List[Document]:

    all_chunks = []

    for file in pdf_dir.glob("*.pdf"):
        # Extract main content
        content = extract_text_from_pdf(file, preserve_layout=True)

        # Extract tables separately if needed
        if handle_tables:
            tables = extract_tables_from_pdf(file)
            content += "\n\n".join(tables)

        # Create documents with metadata
        chunks = chunk_text_with_tokenizer(content, llama_tokenizer) # Use the downloaded tokenizer
        documents = [Document(page_content=chunk, metadata={"source": file.name, "file_type": "pdf"}) for chunk in chunks]
        all_chunks.extend(documents)

    return all_chunks