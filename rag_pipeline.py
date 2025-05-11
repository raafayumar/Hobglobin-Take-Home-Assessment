from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from ollama_config import get_llm_model, get_embedding_model

class RAGPipeline:
    def __init__(self, persist_dir: str = "rag_db"):
        self.embedding_model = get_embedding_model()
        self.vector_store = Chroma(
            collection_name="fine_prints",
            embedding_function=self.embedding_model,
            persist_directory=persist_dir
        )
        self.llm = get_llm_model()
        
        # Define the chat history manually
        self.chat_history = []
        
        # Define the prompt template
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based only on the provided context.

If the answer is not in the context, respond with:
"I couldn't find that information in the documents."

Use only the information below to answer.

Context:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # Set up the retriever
        self.retriever = self.vector_store.as_retriever(search_type="mmr")

        # Create the ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True,
            verbose=False,
            get_chat_history=lambda h: h  # Just pass through the history we provide
        )

    def index_documents(self, docs: List[Document]):
        """Add documents to ChromaDB."""
        self.vector_store.add_documents(docs)

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question and return the answer, source documents, and updated chat history."""
        # Invoke chain with manually provided chat history
        result = self.qa_chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result["answer"]))
        
        return {
            "answer": result["answer"],
            "sources": result["source_documents"],
            "chat_history": self.chat_history
        }
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []