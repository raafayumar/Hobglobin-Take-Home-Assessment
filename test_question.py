# Created by Raafay Umar for Hobglobin Software - AI Engineer Take-Home Assessment on 5th May 2025

"""
This script sends predefined evaluation questions to the `/chat` endpoint.

- Uses HTTP POST requests to simulate user interaction.
- Logs answers and source references to both terminal and `chat_response.txt`.
- Ensures all test results are reproducible and ready for submission.

Used to validate the final RAG setup against key task prompts.
"""

import requests

BASE_URL = "http://127.0.0.1:8000"
OUTPUT_FILE = "chat_response.txt"
FINE_PRINTS_FILE = "fine_prints.txt"


# Define evaluation questions
questions = [
    "List all mandatory documents bidders must submit.",
    "What are the security requirements and site access protocols for the project IFPQ # 01A6494?",
    "What permits or approvals are necessary for this project?",
    "How have you addressed safety challenges specific to public works projects in your past experience?"
]

def ask_question(question: str, output_handle):
    output_handle.write(f"\nQuestion: {question}\n")
    print(f"\nQuestion: {question}")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"query": question}
        )
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            output_handle.write(f"Answer:\n{answer}\n")
            print(f"Answer:\n{answer}")

            if sources:
                output_handle.write("\nSources:\n")
                print("\nSources:")
                for i, source in enumerate(sources, 1):
                    line = f"  {i}. {source['source']}"
                    output_handle.write(line + "\n")
                    print(line)
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            output_handle.write(error_msg + "\n")
            print(error_msg)
    except Exception as e:
        error_msg = f"Exception: {e}"
        output_handle.write(error_msg + "\n")
        print(error_msg)
        
def save_fine_prints():
    print("\nFetching fine-print chunks from /fine-prints...")
    try:
        response = requests.get(f"{BASE_URL}/fine-prints")
        if response.status_code == 200:
            data = response.json()
            chunks = data.get("fine_prints", [])

            with open(FINE_PRINTS_FILE, "w", encoding="utf-8") as f:
                f.write("# fine_prints.txt\n\n")
                f.write("This file contains all the extracted fine-print chunks from the PDF documents that were indexed into the RAG system.\n\n")
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"--- Fine Print #{i} ---\n")
                    f.write(chunk.strip() + "\n\n")
            print(f"Fine-print chunks saved to {FINE_PRINTS_FILE}")
        else:
            print(f"Failed to fetch fine-prints: {response.status_code}")
    except Exception as e:
        print(f"Exception while saving fine_prints.txt: {e}")

if __name__ == "__main__":
    print("Hobglobin RAG Chat Evaluation\n")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Hobglobin RAG Chat Evaluation - Output Log\n")
        f.write("="*60 + "\n")
        for q in questions:
            ask_question(q, f)

    print(f"\nAll responses saved to {OUTPUT_FILE}")
    save_fine_prints()
