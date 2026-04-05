"""
RAG Evaluation with RAGAS and LiteLLM

This script demonstrates:
- Evaluating RAG systems with RAGAS metrics
- Comparing different LLM providers/configurations
- Measuring faithfulness, answer relevancy, and context precision
- Using LiteLLM for evaluation LLM calls
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


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
            metadata={'doc_id': doc_data['id'], 'title': doc_data['title']}
        )
        documents.append(doc)
    
    return documents


def load_eval_dataset(data_path: str = "sample_data/eval_dataset.json") -> List[Dict]:
    """Load evaluation dataset."""
    script_dir = Path(__file__).parent
    file_path = script_dir / data_path
    
    with open(file_path, 'r') as f:
        eval_data = json.load(f)
    
    print(f"✅ Loaded {len(eval_data)} evaluation questions")
    return eval_data


def create_rag_index(
    documents: List[Document],
    model_name: str,
    collection_name: str = "ragas_eval"
) -> VectorStoreIndex:
    """Create RAG index with specified model."""
    llm = LiteLLM(
        model=model_name,
        temperature=0.1,
        max_tokens=512,
        num_retries=3,
        timeout=30.0,
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        pooling="cls",
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # Create vector store
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=False,
    )
    
    return index


def generate_rag_responses(
    index: VectorStoreIndex,
    eval_dataset: List[Dict],
    top_k: int = 3
) -> Dict[str, List]:
    """
    Generate RAG responses for evaluation dataset.
    
    Returns data in RAGAS format:
    - question: User question
    - answer: Generated answer
    - contexts: Retrieved document chunks
    - ground_truth: Expected answer
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",
    )
    
    print(f"🔄 Generating responses for {len(eval_dataset)} questions...")
    
    for i, item in enumerate(eval_dataset, 1):
        question = item['question']
        ground_truth = item['ground_truth']
        
        # Query RAG system
        response = query_engine.query(question)
        answer = str(response.response)
        
        # Extract contexts
        retrieved_contexts = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                retrieved_contexts.append(node.text)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)
        
        print(f"  ✓ Processed {i}/{len(eval_dataset)}: {question[:60]}...")
    
    return {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths,
    }


def evaluate_rag_system(
    rag_data: Dict[str, List],
    model_name: str,
    eval_model: str = "gpt-3.5-turbo"
) -> Dict:
    """
    Evaluate RAG system using RAGAS metrics.
    
    Metrics:
    - Faithfulness: Is answer grounded in retrieved context?
    - Answer Relevancy: Does answer address the question?
    - Context Precision: Are retrieved docs relevant?
    - Context Recall: Was all needed info retrieved?
    """
    print(f"\n📊 Evaluating with RAGAS (using {eval_model} as judge)...")
    
    # Require API key explicitly — do not silently fall back to a dummy value
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "RAGAS requires an LLM to act as a judge. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    # Convert to RAGAS dataset format
    dataset = Dataset.from_dict(rag_data)
    
    # Wire eval_model into RAGAS via LangchainLLMWrapper so the user-specified
    # judge model is actually used instead of the RAGAS default.
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    eval_llm = LangchainLLMWrapper(ChatOpenAI(model=eval_model))
    
    # Run evaluation
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=eval_llm,
        )
        
        scores = {
            'model': model_name,
            'faithfulness': result['faithfulness'],
            'answer_relevancy': result['answer_relevancy'],
            'context_precision': result['context_precision'],
            'context_recall': result['context_recall'],
            'overall_score': (
                result['faithfulness'] + 
                result['answer_relevancy'] + 
                result['context_precision'] + 
                result['context_recall']
            ) / 4
        }
        
        return scores
    
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return {
            'model': model_name,
            'error': str(e),
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'context_precision': 0.0,
            'context_recall': 0.0,
            'overall_score': 0.0,
        }


def compare_models(
    documents: List[Document],
    eval_dataset: List[Dict],
    models: List[str]
) -> List[Dict]:
    """
    Compare multiple models/providers using RAGAS evaluation.
    
    This demonstrates how LiteLLM enables easy A/B testing
    across different providers and model configurations.
    """
    results = []
    
    for model in models:
        print("\n" + "=" * 70)
        print(f"Evaluating Model: {model}")
        print("=" * 70)
        
        try:
            # Create RAG index with this model
            index = create_rag_index(
                documents,
                model_name=model,
                collection_name=f"eval_{model.replace('/', '_')}"
            )
            
            # Generate responses
            rag_data = generate_rag_responses(index, eval_dataset, top_k=3)
            
            # Evaluate
            scores = evaluate_rag_system(rag_data, model_name=model)
            results.append(scores)
            
            # Print results
            print(f"\n✅ Results for {model}:")
            print(f"   Faithfulness:      {scores['faithfulness']:.3f}")
            print(f"   Answer Relevancy:  {scores['answer_relevancy']:.3f}")
            print(f"   Context Precision: {scores['context_precision']:.3f}")
            print(f"   Context Recall:    {scores['context_recall']:.3f}")
            print(f"   Overall Score:     {scores['overall_score']:.3f}")
        
        except Exception as e:
            print(f"❌ Error evaluating {model}: {e}")
            results.append({
                'model': model,
                'error': str(e),
                'overall_score': 0.0
            })
    
    return results


def print_comparison_table(results: List[Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("RAGAS Evaluation Comparison")
    print("=" * 70)
    print(f"{'Model':<30} {'Faith':<8} {'Relev':<8} {'Prec':<8} {'Recall':<8} {'Overall':<8}")
    print("-" * 70)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['model']:<30} "
                  f"{result['faithfulness']:<8.3f} "
                  f"{result['answer_relevancy']:<8.3f} "
                  f"{result['context_precision']:<8.3f} "
                  f"{result['context_recall']:<8.3f} "
                  f"{result['overall_score']:<8.3f}")
        else:
            print(f"{result['model']:<30} ERROR: {result['error']}")
    
    print("=" * 70)


def main():
    """Main execution flow."""
    print("=" * 70)
    print("RAG Evaluation with RAGAS and LiteLLM")
    print("=" * 70 + "\n")
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set.")
        print("   RAGAS requires an LLM for evaluation metrics.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'\n")
        return
    
    # Load data
    documents = load_documents()
    eval_dataset = load_eval_dataset()
    
    # Models to compare
    # Uncomment models you have API keys for
    models_to_test = [
        "gpt-3.5-turbo",
        # "gpt-4-turbo",
        # "claude-3-haiku-20240307",
        # "groq/llama3-70b-8192",
    ]
    
    print(f"\n🔬 Testing {len(models_to_test)} model configurations...")
    
    # Run comparison
    results = compare_models(documents, eval_dataset, models_to_test)
    
    # Print comparison
    print_comparison_table(results)
    
    # Save results
    output_path = Path(__file__).parent / "ragas_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Detailed results saved to: {output_path}")
    
    # Insights
    print("\n💡 Key Insights:")
    print("  - RAGAS uses LLMs as judges for evaluation")
    print("  - Faithfulness measures hallucination (higher is better)")
    print("  - Answer Relevancy checks if answer addresses question")
    print("  - Context metrics evaluate retrieval quality")
    print("  - LiteLLM makes it easy to compare providers")
    print("  - Add more models to models_to_test list for comparison")
    
    print("\n" + "=" * 70)
    print("🎉 Evaluation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()