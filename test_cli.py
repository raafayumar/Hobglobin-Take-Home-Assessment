# Created by Raafay Umar for Hobglobin Software - AI Engineer Take-Home Assessment on 5th May 2025

"""
This CLI script allows local testing of the RAG pipeline.

- Loads PDFs, indexes them, and enters an interactive Q&A loop.
- Supports commands like 'exit', 'clear', and toggling source display.
- Prints nicely formatted answers and chunks from source documents.

Intended for manual testing before deploying the API.
"""


from pathlib import Path
from pdf_parser import load_all_pdfs
from rag_pipeline import RAGPipeline
import textwrap

def print_wrapped(text, width=80, indent=0):
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=' '*indent)
    print(wrapper.fill(text))

def display_sources(sources):
    print("\nSources:")
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"\nSource {i}: {source} (page {page})")
        print("-"*50)
        print_wrapped(doc.page_content, indent=2)
        print()

def main():
    # Load and parse PDFs
    pdf_dir = Path("data/pdfs")
    print("Loading and parsing PDFs...")
    documents = load_all_pdfs(pdf_dir)
    
    # Build RAG pipeline and index documents
    print("Building RAG pipeline and indexing documents...")
    rag = RAGPipeline()
    rag.index_documents(documents)
    print("\nPDFs indexed. You can now ask questions.")
    print("Type 'exit' to quit, 'clear' to reset conversation, or 'sources' to toggle source display\n")

    # Conversation settings
    show_sources = False

    # CLI loop
    while True:
        try:
            query = input("\nYou: ")
            
            if query.lower() == 'exit':
                break
                
            if query.lower() == 'clear':
                rag.clear_history()
                print("\nConversation history cleared")
                continue
                
            if query.lower() == 'sources':
                show_sources = not show_sources
                status = "ON" if show_sources else "OFF"
                print(f"\nSource display {status}")
                continue

            print("\nThinking...")
                            
            # Process query
            result = rag.query(query)

            
            # Display response
            print("\nAnswer:")
            print_wrapped(result['answer'])
            
            # Display sources if enabled
            if show_sources and result['sources']:
                display_sources(result['sources'])
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

    print("\nbye!")

if __name__ == "__main__":
    main()