# LiteLLM RAG Gateway Cookbook

A comprehensive example demonstrating production-ready RAG (Retrieval-Augmented Generation) systems using **LiteLLM** as a multi-provider LLM gateway, integrated with **LlamaIndex**, **Haystack**, and evaluated with **RAGAS**.

## 🎯 What This Example Demonstrates

This cookbook showcases:

- **Multi-Provider LLM Gateway**: Use LiteLLM to seamlessly switch between OpenAI, Anthropic, Groq, Ollama, and 100+ other providers
- **Production-Ready Patterns**: Automatic retries, fallback chains, timeouts, and comprehensive logging
- **RAG with LlamaIndex**: Build semantic search pipelines with vector embeddings and ChromaDB
- **RAG with Haystack**: Create modular, component-based RAG pipelines
- **Evaluation with RAGAS**: Measure faithfulness, answer relevancy, context precision, and recall
- **Provider Comparison**: A/B test different models and providers with consistent code

## 📁 Project Structure

```
rag_gateway/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── litellm_config.yaml           # Multi-provider configuration (optional)
├── rag_llamaindex.py             # LlamaIndex RAG pipeline
├── rag_haystack.py               # Haystack RAG pipeline
├── evaluate_with_ragas.py        # RAGAS evaluation script
└── sample_data/
    ├── documents.json            # Knowledge base (10 AI/ML docs)
    └── eval_dataset.json         # Evaluation questions & ground truth
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+ (tested with 3.12)
- API keys for LLM providers (at minimum, OpenAI)

### 2. Installation

```bash
# Navigate to this directory
cd cookbook/rag_gateway

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set API Keys

```bash
# Required for most examples
export OPENAI_API_KEY="your-openai-key-here"

# Optional: For testing alternative providers
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export GROQ_API_KEY="your-groq-key-here"

# For local models (free, no API key needed)
# Install Ollama: https://ollama.ai
# ollama pull llama3
```

### 4. Run Examples

#### LlamaIndex RAG Pipeline

```bash
python rag_llamaindex.py
```

**What it does:**
- Loads 10 AI/ML documents into memory
- Creates vector embeddings using HuggingFace models
- Stores embeddings in ChromaDB (in-memory)
- Runs 3 example queries with semantic search
- Saves results to `llamaindex_results.json`

**Expected output:**
```
======================================================================
LlamaIndex RAG with LiteLLM Multi-Provider Gateway
======================================================================

✅ Loaded 10 documents
✅ Configured LiteLLM with model: gpt-3.5-turbo
✅ Configured HuggingFace embeddings
🔄 Creating vector index (this may take a moment)...
✅ RAG pipeline created successfully

❓ Question: What are the main components of RAG architecture?

📚 Retrieved Sources:
  1. RAG Architecture and Benefits (score: 0.892)
     Preview: Retrieval-Augmented Generation (RAG) combines...

💡 Answer: The main components of RAG architecture are...
```

#### Haystack RAG Pipeline

```bash
python rag_haystack.py
```

**What it does:**
- Demonstrates Haystack's component-based pipeline architecture
- Uses custom LiteLLM generator component
- Shows how to integrate LiteLLM with Haystack 2.0
- Saves results to `haystack_results.json`

#### RAGAS Evaluation

```bash
python evaluate_with_ragas.py
```

**What it does:**
- Loads evaluation dataset with 8 questions
- Generates answers using RAG pipeline
- Evaluates with RAGAS metrics (faithfulness, relevancy, precision, recall)
- Compares multiple models/providers
- Saves detailed results to `ragas_evaluation_results.json`

**Expected output:**
```
======================================================================
RAGAS Evaluation Comparison
======================================================================
Model                          Faith    Relev    Prec     Recall   Overall
----------------------------------------------------------------------
gpt-3.5-turbo                  0.892    0.945    0.878    0.901    0.904
gpt-4-turbo                    0.934    0.967    0.912    0.945    0.940
======================================================================
```

## 🔧 Switching LLM Providers

One of LiteLLM's key benefits is **zero-code provider switching**. Simply change the `model_name` parameter:

### In Python Scripts

```python
# OpenAI
llm = setup_litellm(model_name="gpt-3.5-turbo")
llm = setup_litellm(model_name="gpt-4-turbo")

# Anthropic Claude
llm = setup_litellm(model_name="claude-3-haiku-20240307")
llm = setup_litellm(model_name="claude-3-sonnet-20240229")

# Groq (ultra-fast inference)
llm = setup_litellm(model_name="groq/llama3-70b-8192")
llm = setup_litellm(model_name="groq/mixtral-8x7b-32768")

# Local Ollama (free, private)
llm = setup_litellm(model_name="ollama/llama3")
llm = setup_litellm(model_name="ollama/mistral")

# Azure OpenAI
llm = setup_litellm(model_name="azure/gpt-4-deployment-name")
```

### Using Configuration File (Optional)

The `litellm_config.yaml` file demonstrates advanced configuration:

```yaml
model_list:
  - model_name: gpt-4-turbo
    litellm_params:
      model: gpt-4-turbo-preview
      api_key: os.environ/OPENAI_API_KEY
      timeout: 30
      max_retries: 3

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
  fallbacks:
    - gpt-4-turbo
    - gpt-3.5-turbo
    - claude-3-haiku
```

## 📊 Understanding RAGAS Metrics

RAGAS evaluates RAG systems using LLM-based metrics:

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (hallucination check) | > 0.8 |
| **Answer Relevancy** | Does the answer address the question? | > 0.8 |
| **Context Precision** | Are retrieved documents relevant to the question? | > 0.7 |
| **Context Recall** | Was all necessary information retrieved? | > 0.7 |

Higher scores are better. RAGAS uses an LLM as a judge, making evaluation more aligned with human judgment than traditional metrics like BLEU or ROUGE.

## 🎓 Key Concepts

### Why LiteLLM?

1. **Unified Interface**: One API for 100+ providers
2. **Reliability**: Built-in retries, fallbacks, timeouts
3. **Cost Tracking**: Monitor spending per request
4. **Easy Testing**: Switch providers without code changes
5. **Production Ready**: Used by companies for high-availability systems

### RAG Architecture

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector Database] → Retrieve Top-K Documents
    ↓
[LLM via LiteLLM] → Generate Answer from Context
    ↓
Answer + Sources
```

### Production Best Practices Demonstrated

- ✅ **Retry Logic**: Automatic retries with exponential backoff
- ✅ **Fallback Chains**: Switch to backup providers on failure
- ✅ **Timeouts**: Prevent hanging requests
- ✅ **Logging**: Track all requests for debugging
- ✅ **Cost Tracking**: Monitor LLM API costs
- ✅ **Evaluation**: Measure quality with RAGAS
- ✅ **Source Attribution**: Return retrieved documents
- ✅ **Error Handling**: Graceful degradation

## 🔍 Sample Data

The `sample_data/` directory contains:

### documents.json
10 comprehensive documents covering:
- Large Language Models
- RAG Architecture
- LiteLLM Features
- Vector Embeddings
- LlamaIndex & Haystack
- RAGAS Evaluation
- Production Best Practices
- Prompt Engineering
- Cost Optimization

### eval_dataset.json
8 evaluation questions with:
- Question text
- Ground truth answer
- Expected source document IDs

Perfect for testing and benchmarking your RAG system.

## 🛠️ Customization

### Add Your Own Documents

Edit `sample_data/documents.json`:

```json
[
  {
    "id": "doc_11",
    "title": "Your Document Title",
    "content": "Your document content here..."
  }
]
```

### Change Embedding Models

In the scripts, modify:

```python
# Use different HuggingFace model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"  # Larger, more accurate
)

# Or use OpenAI embeddings
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
```

### Adjust Retrieval Parameters

```python
# Retrieve more documents
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Default is 3
    response_mode="tree_summarize"  # Different synthesis strategy
)
```

### Add More Models to Compare

In `evaluate_with_ragas.py`:

```python
models_to_test = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "groq/llama3-70b-8192",
    "ollama/llama3",
]
```

## 🐛 Troubleshooting

### "API key not found"

```bash
# Make sure you've exported your API key
echo $OPENAI_API_KEY

# If empty, set it:
export OPENAI_API_KEY="sk-..."
```

### "Module not found" errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Slow embedding generation

First run downloads HuggingFace models (~100MB). Subsequent runs are fast.

### RAGAS evaluation fails

RAGAS requires an LLM for evaluation. Ensure `OPENAI_API_KEY` is set.

### Ollama connection refused

```bash
# Start Ollama server
ollama serve

# In another terminal, pull a model
ollama pull llama3
```

## 📚 Learn More

- **LiteLLM Docs**: https://docs.litellm.ai/
- **LlamaIndex Docs**: https://docs.llamaindex.ai/
- **Haystack Docs**: https://docs.haystack.deepset.ai/
- **RAGAS Docs**: https://docs.ragas.io/

## 🤝 Contributing

This example is part of the LiteLLM project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This example follows the LiteLLM project license.

## 🎉 What's Next?

After running this example, you can:

1. **Deploy to Production**: Use LiteLLM Proxy for authentication, rate limiting, and load balancing
2. **Scale Up**: Replace in-memory vector stores with Pinecone, Weaviate, or Qdrant
3. **Add Caching**: Implement semantic caching to reduce API costs
4. **Fine-tune Retrieval**: Experiment with hybrid search (semantic + keyword)
5. **Monitor Performance**: Integrate with Langfuse or other observability tools
6. **Build an API**: Wrap your RAG pipeline in FastAPI or Flask
7. **Add Streaming**: Enable streaming responses for better UX

---

**Built with ❤️ to demonstrate LiteLLM's power for production RAG systems**
