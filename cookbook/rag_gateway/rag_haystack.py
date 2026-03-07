"""
Haystack RAG Pipeline with LiteLLM Multi-Provider Gateway

This script demonstrates:
- Building a RAG pipeline with Haystack 2.0
- Using LiteLLM for flexible LLM provider switching
- Document ingestion and retrieval
- Answer generation with source attribution
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from haystack import Pipeline, Document, component
from haystack.components.builders import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
import litellm


def load_documents(data_path: str = "sample_data/documents.json") -> List[Document]:
    """Load documents from JSON and convert to Haystack format."""
    script_dir = Path(__file__).parent
    file_path = script_dir / data_path
    
    with open(file_path, 'r') as f:
        docs_data = json.load(f)
    
    documents = []
    for doc_data in docs_data:
        doc = Document(
            content=doc_data['content'],
            meta={
                'doc_id': doc_data['id'],
                'title': doc_data['title']
            }
        )
        documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} documents")
    return documents


@component
class LiteLLMGenerator:
    """
    Custom Haystack component that uses LiteLLM for generation.
    
    This allows seamless switching between providers (OpenAI, Anthropic, etc.)
    while maintaining Haystack's pipeline architecture.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 512,
        num_retries: int = 3,
        timeout: float = 30.0
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.timeout = timeout
        
        # Configure LiteLLM settings
        litellm.num_retries = num_retries
        litellm.request_timeout = timeout
        
        print(f"✅ Configured LiteLLM Generator with model: {model}")
    
    @component.output_types(replies=List[str], meta=Dict[str, Any])
    def run(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response using LiteLLM.
        
        LiteLLM handles:
        - Provider-specific API translation
        - Automatic retries with exponential backoff
        - Fallback to alternative providers
        - Cost tracking and logging
        """
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                num_retries=self.num_retries,
                timeout=self.timeout,
                stream=False,  # Explicitly disable streaming
                # Optional: Enable fallbacks
                fallbacks=["gpt-3.5-turbo", "claude-3-haiku-20240307"],
            )
            
            # Safely extract answer
            answer = ""
            if hasattr(response, 'choices') and len(response.choices) > 0:  # type: ignore
                answer = response.choices[0].message.content  # type: ignore
            else:
                answer = str(response)
            
            # Safely extract metadata
            meta: Dict[str, Any] = {"model": self.model}
            if hasattr(response, 'model') and response.model:
                meta["model"] = str(response.model)
            if hasattr(response, 'usage') and response.usage:  # type: ignore
                meta["usage"] = {
                    "prompt_tokens": int(response.usage.prompt_tokens),  # type: ignore
                    "completion_tokens": int(response.usage.completion_tokens),  # type: ignore
                    "total_tokens": int(response.usage.total_tokens),  # type: ignore
                }
            
            return {
                "replies": [answer],
                "meta": meta
            }
        
        except Exception as e:
            print(f"❌ Error in LiteLLM generation: {e}")
            return {
                "replies": [f"Error: {str(e)}"],
                "meta": {"error": str(e)}
            }


def create_document_store_with_embeddings(documents: List[Document]) -> InMemoryDocumentStore:
    """
    Create document store and add embeddings for semantic search.
    """
    # Initialize document store
    document_store = InMemoryDocumentStore()
    
    # Create embedder
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()
    
    # Generate embeddings
    print("🔄 Generating embeddings for documents...")
    docs_with_embeddings = doc_embedder.run(documents)
    
    # Write to store
    document_store.write_documents(docs_with_embeddings["documents"])
    
    print(f"✅ Document store created with {document_store.count_documents()} documents")
    return document_store


def create_rag_pipeline(
    document_store: InMemoryDocumentStore,
    model_name: str = "gpt-3.5-turbo"
) -> Pipeline:
    """
    Create Haystack RAG pipeline with LiteLLM generator.
    
    Pipeline components:
    1. Text Embedder - Convert query to embedding
    2. Retriever - Fetch relevant documents
    3. Prompt Builder - Format context and question
    4. LiteLLM Generator - Generate answer
    """
    # Initialize components
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    
    # RAG prompt template
    template = """
You are a helpful AI assistant. Answer the question based on the provided context.
If the context doesn't contain enough information, say so.

Context:
{% for document in documents %}
  {{ document.content }}
{% endfor %}

Question: {{ question }}

Answer:
"""
    
    prompt_builder = PromptBuilder(template=template)
    
    llm_generator = LiteLLMGenerator(model=model_name)
    
    # Build pipeline
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", text_embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm_generator)
    
    # Connect components
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    
    print("✅ Haystack RAG pipeline created successfully")
    return pipeline


def query_rag(
    pipeline: Pipeline,
    question: str,
    top_k: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Query the RAG pipeline and return answer with sources.
    
    Args:
        pipeline: Haystack pipeline
        question: User question
        top_k: Number of documents to retrieve
        verbose: Print retrieved context
    
    Returns:
        Dict with answer, sources, and metadata
    """
    print(f"\n❓ Question: {question}")
    
    # Run pipeline
    result = pipeline.run({
        "text_embedder": {"text": question},
        "retriever": {"top_k": top_k},
        "prompt_builder": {"question": question}
    })
    
    # Extract answer
    answer = result["llm"]["replies"][0]
    metadata = result["llm"].get("meta", {})
    
    # Extract sources
    sources = []
    if "retriever" in result and "documents" in result["retriever"]:
        for doc in result["retriever"]["documents"]:
            sources.append({
                'doc_id': doc.meta.get('doc_id', 'unknown'),
                'title': doc.meta.get('title', 'unknown'),
                'score': doc.score if hasattr(doc, 'score') else 0.0,
                'text_preview': doc.content[:200] + "..."
            })
    
    if verbose and sources:
        print("\n📚 Retrieved Sources:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source['title']} (score: {source['score']:.3f})")
            print(f"     Preview: {source['text_preview']}\n")
    
    print(f"💡 Answer: {answer}\n")
    
    if verbose and metadata.get('usage'):
        usage = metadata['usage']
        print(f"📊 Token Usage: {usage['total_tokens']} total "
              f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)")
    
    return {
        'question': question,
        'answer': answer,
        'sources': sources,
        'metadata': metadata
    }


def main():
    """Main execution flow."""
    print("=" * 70)
    print("Haystack RAG with LiteLLM Multi-Provider Gateway")
    print("=" * 70 + "\n")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Set it with:")
        print("   export OPENAI_API_KEY='your-key-here'\n")
        print("   Attempting to use fallback providers...\n")
    
    # Load documents
    documents = load_documents()
    
    # Create document store with embeddings
    document_store = create_document_store_with_embeddings(documents)
    
    # Create RAG pipeline
    # Try different models: "gpt-4-turbo", "gpt-3.5-turbo", 
    # "claude-3-haiku-20240307", "groq/llama3-70b-8192"
    pipeline = create_rag_pipeline(document_store, model_name="gpt-3.5-turbo")
    
    # Example queries
    questions = [
        "What are the key differences between LlamaIndex and Haystack?",
        "How does LiteLLM provide reliability for production systems?",
        "What are best practices for chunking documents in RAG?",
    ]
    
    results = []
    for question in questions:
        result = query_rag(pipeline, question, top_k=3)
        results.append(result)
        print("-" * 70 + "\n")
    
    # Save results
    output_path = Path(__file__).parent / "haystack_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("🎉 Haystack RAG pipeline completed successfully!")
    print("=" * 70)
    
    # Tips
    print("\n💡 Tips:")
    print("  - Haystack's component architecture makes pipelines modular")
    print("  - Switch LLM providers by changing model_name parameter")
    print("  - LiteLLM handles retries, fallbacks, and cost tracking")
    print("  - Combine with BM25 retriever for hybrid search")
    print("  - Add re-rankers for improved retrieval quality")


if __name__ == "__main__":
    main()