"""
LlamaIndex RAG Pipeline with LiteLLM Multi-Provider Gateway

This script demonstrates:
- Building a RAG pipeline with LlamaIndex
- Using LiteLLM for multi-provider LLM access
- Configuring retries, fallbacks, and timeouts
- Vector-based semantic search with ChromaDB
"""

import os
import json
from pathlib import Path
from typing import List, Dict

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def load_documents(data_path: str = "sample_data/documents.json") -> List[Document]:
    """Load documents from JSON file."""
    script_dir = Path(__file__).parent
    file_path = script_dir / data_path
    
    with open(file_path, 'r') as f:
        docs_data = json.load(f)
    
    documents = []
    for doc_data in docs_data:
        doc = Document(
            text=doc_data['content'],
            metadata={
                'doc_id': doc_data['id'],
                'title': doc_data['title']
            }
        )
        documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} documents")
    return documents


def setup_litellm(
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.1,
    max_tokens: int = 512
) -> LiteLLM:
    """
    Configure LiteLLM with retry and fallback settings.
    
    LiteLLM automatically handles:
    - Provider-specific API format translation
    - Retries with exponential backoff
    - Fallback to alternative providers
    - Cost tracking and logging
    """
    llm = LiteLLM(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        # These settings enable production-ready reliability
        num_retries=3,  # Retry failed requests
        timeout=30.0,   # 30 second timeout
        # Fallback models (if primary fails)
        fallbacks=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
    )
    
    print(f"✅ Configured LiteLLM with model: {model_name}")
    return llm


def setup_embeddings() -> HuggingFaceEmbedding:
    """Configure embedding model for semantic search."""
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        # Use 'cls' pooling for better performance
        pooling="cls",
    )
    
    print("✅ Configured HuggingFace embeddings")
    return embed_model


def create_rag_pipeline(
    documents: List[Document],
    llm: LiteLLM,
    embed_model: HuggingFaceEmbedding,
    collection_name: str = "litellm_rag_docs"
) -> VectorStoreIndex:
    """
    Create RAG pipeline with vector store and query engine.
    
    Pipeline steps:
    1. Chunk documents into smaller pieces
    2. Generate embeddings for each chunk
    3. Store in ChromaDB vector database
    4. Create query engine for retrieval + generation
    """
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # Initialize ChromaDB
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Parse documents into nodes (chunks)
    node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    
    # Create index with embeddings
    print("🔄 Creating vector index (this may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        node_parser=node_parser,
        show_progress=True,
    )
    
    print("✅ RAG pipeline created successfully")
    return index


def query_rag(
    index: VectorStoreIndex,
    question: str,
    top_k: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Query the RAG system and return answer with sources.
    
    Args:
        index: Vector store index
        question: User question
        top_k: Number of documents to retrieve
        verbose: Print retrieved context
    
    Returns:
        Dict with answer, sources, and metadata
    """
    # Create query engine with custom prompt
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",  # Combine context efficiently
    )
    
    # Execute query
    print(f"\n❓ Question: {question}")
    response = query_engine.query(question)
    
    # Extract source information
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            sources.append({
                'doc_id': node.metadata.get('doc_id', 'unknown'),
                'title': node.metadata.get('title', 'unknown'),
                'score': node.score,
                'text_preview': node.text[:200] + "..."
            })
    
    if verbose and sources:
        print("\n📚 Retrieved Sources:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['title']} (score: {source['score']:.3f})")
            print(f"     Preview: {source['text_preview']}\n")
    
    print(f"💡 Answer: {response.response}\n")
    
    return {
        'question': question,
        'answer': str(response.response),
        'sources': sources,
        'metadata': {
            'model': Settings.llm.model,
            'top_k': top_k
        }
    }


def main():
    """Main execution flow."""
    print("=" * 70)
    print("LlamaIndex RAG with LiteLLM Multi-Provider Gateway")
    print("=" * 70 + "\n")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Set it with:")
        print("   export OPENAI_API_KEY='your-key-here'\n")
        print("   Attempting to use fallback providers...\n")
    
    # Load documents
    documents = load_documents()
    
    # Setup LiteLLM (try different models by changing model_name)
    # Options: "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-haiku-20240307", 
    #          "groq/llama3-70b-8192", "ollama/llama3"
    llm = setup_litellm(model_name="gpt-3.5-turbo")
    
    # Setup embeddings
    embed_model = setup_embeddings()
    
    # Create RAG pipeline
    index = create_rag_pipeline(documents, llm, embed_model)
    
    # Example queries
    questions = [
        "What are the main components of RAG architecture?",
        "How does LiteLLM help with multi-provider integration?",
        "What metrics does RAGAS use for evaluation?",
    ]
    
    results = []
    for question in questions:
        result = query_rag(index, question, top_k=3)
        results.append(result)
        print("-" * 70 + "\n")
    
    # Save results
    output_path = Path(__file__).parent / "llamaindex_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("🎉 LlamaIndex RAG pipeline completed successfully!")
    print("=" * 70)
    
    # Tips for switching providers
    print("\n💡 Tips:")
    print("  - Change model by modifying 'model_name' parameter")
    print("  - Set ANTHROPIC_API_KEY to use Claude models")
    print("  - Set GROQ_API_KEY for ultra-fast inference")
    print("  - Install Ollama for free local models")
    print("  - LiteLLM automatically handles retries and fallbacks")


if __name__ == "__main__":
    main()