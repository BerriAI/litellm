# Benchmark: Semantic Code Search (claude-context) vs. Grep-based Search

## Overview

This benchmark compares two code retrieval approaches for answering natural language questions about the LiteLLM codebase (~1800 Python files, ~125K LOC):

1. **Semantic Search** (claude-context style): `all-MiniLM-L6-v2` sentence-transformer embeddings + Milvus Lite vector DB with cosine similarity. Code split into semantic chunks using Python AST parsing.
2. **Grep-based Search**: ripgrep keyword search with context lines, file ranking by match density, and targeted snippet extraction.

Both approaches feed retrieved context to `claude-sonnet-4-20250514` to generate answers. Answers are scored by `claude-sonnet-4-20250514` as judge against human-written ground truth.

## Setup Details

| Parameter | Value |
|---|---|
| Codebase | LiteLLM (~1800 .py files, ~125K LOC) |
| Chunks indexed (semantic) | 13561 |
| Embedding model | all-MiniLM-L6-v2 (local, 384-dim) |
| Embedding time (one-time) | 132.3s |
| Answering model | claude-sonnet-4-20250514 |
| Judge model | claude-sonnet-4-20250514 |
| Max context tokens | 12000 |
| Number of queries | 10 |

## Per-Query Results

| Query | Category | Approach | Accuracy | Completeness | Specificity | Relevance | Total (/20) | Search Time | Context Tokens |
|---|---|---|---|---|---|---|---|---|---|
| q1 | architecture | Semantic | 4 | 5 | 5 | 4 | 18 | 0.03s | 3780 |
| q1 | architecture | Grep | 5 | 5 | 5 | 5 | 20 | 0.14s | 12006 |
| q2 | architecture | Semantic | 5 | 5 | 5 | 5 | 20 | 0.03s | 3471 |
| q2 | architecture | Grep | 5 | 5 | 5 | 5 | 20 | 0.12s | 8801 |
| q3 | proxy | Semantic | 5 | 5 | 5 | 5 | 20 | 0.02s | 5601 |
| q3 | proxy | Grep | 5 | 5 | 5 | 5 | 20 | 0.19s | 12006 |
| q4 | core | Semantic | 4 | 4 | 5 | 5 | 18 | 0.02s | 2481 |
| q4 | core | Grep | 5 | 4 | 5 | 5 | 19 | 0.08s | 12012 |
| q5 | feature | Semantic | 4 | 5 | 5 | 5 | 19 | 0.02s | 4517 |
| q5 | feature | Grep | 5 | 5 | 5 | 5 | 20 | 0.12s | 12013 |
| q6 | provider | Semantic | 3 | 2 | 3 | 4 | 12 | 0.01s | 1355 |
| q6 | provider | Grep | 5 | 5 | 5 | 5 | 20 | 0.13s | 12004 |
| q7 | proxy | Semantic | 5 | 5 | 5 | 5 | 20 | 0.01s | 3084 |
| q7 | proxy | Grep | 4 | 5 | 5 | 5 | 19 | 0.14s | 12005 |
| q8 | infrastructure | Semantic | 4 | 3 | 5 | 4 | 16 | 0.02s | 3998 |
| q8 | infrastructure | Grep | 4 | 5 | 5 | 5 | 19 | 0.10s | 10910 |
| q9 | reliability | Semantic | 5 | 5 | 5 | 5 | 20 | 0.02s | 5012 |
| q9 | reliability | Grep | 5 | 5 | 5 | 5 | 20 | 0.13s | 12009 |
| q10 | observability | Semantic | 5 | 5 | 5 | 5 | 20 | 0.02s | 3381 |
| q10 | observability | Grep | 5 | 5 | 5 | 5 | 20 | 0.13s | 12006 |

## Aggregate Scores

| Metric | Semantic (avg) | Grep (avg) | Winner |
|---|---|---|---|
| Accuracy | 4.40 | 4.80 | Grep |
| Completeness | 4.40 | 4.90 | Grep |
| Specificity | 4.80 | 5.00 | Grep |
| Relevance | 4.70 | 5.00 | Grep |
| **Total (/20)** | 18.30 | 19.70 | Grep |

## Performance Comparison

| Metric | Semantic (avg) | Grep (avg) |
|---|---|---|
| Search time | 0.02s | 0.13s |
| Answer gen time | 16.91s | 20.18s |
| Total time per query | 16.93s | 20.31s |
| Context tokens (avg) | 3668 | 11577 |
| One-time indexing cost | 132.3s | 0s |

## Head-to-Head

| Outcome | Count |
|---|---|
| Semantic wins | 1/10 |
| Grep wins | 5/10 |
| Ties | 4/10 |

## Performance by Category

| Category | Semantic Avg Total | Grep Avg Total | Winner |
|---|---|---|---|
| architecture | 19.0 | 20.0 | Grep |
| core | 18.0 | 19.0 | Grep |
| feature | 19.0 | 20.0 | Grep |
| infrastructure | 16.0 | 19.0 | Grep |
| observability | 20.0 | 20.0 | Tie |
| provider | 12.0 | 20.0 | Grep |
| proxy | 20.0 | 19.5 | Semantic |
| reliability | 20.0 | 20.0 | Tie |

## Key Findings

**Overall winner: Grep-based Search** (margin: 1.40/20)

### Semantic Search (claude-context style)
**Strengths:**
- Understands natural language queries semantically
- Finds conceptually related code even without exact keyword matches
- Returns focused, relevant code chunks (AST-aware splitting)
- Consistent retrieval quality regardless of query phrasing

**Weaknesses:**
- Requires upfront indexing (132s for 13561 chunks)
- Each search requires an embedding computation
- May miss exact symbol matches if embedding doesn't capture them
- Requires API keys for embedding model + vector database (in cloud mode)

### Grep-based Search
**Strengths:**
- Zero indexing overhead
- Exact matching for known symbols
- Fast per-query search (sub-second, no API call)
- No external dependencies for search
- Returns exact line-level matches with surrounding context

**Weaknesses:**
- Requires knowing the right keywords
- Cannot understand semantic intent
- May return irrelevant matches for common terms
- Context extraction is heuristic-based

## Methodology Notes

- **Semantic search** mirrors claude-context's approach: AST-based code chunking, 
  dense embeddings, vector similarity search via Milvus. Uses local `all-MiniLM-L6-v2`
  instead of OpenAI `text-embedding-3-small` (claude-context default).
  Note: claude-context also uses BM25 hybrid search which we approximate with dense-only.
- **Grep search** mirrors typical coding agent behavior: keyword-based ripgrep, 
  file ranking, targeted snippet extraction.
- Both use the same context budget (12K tokens) and answering model (Claude).
- Scoring: Claude as judge, 4 dimensions x 5 points = 20 max.
- The grep approach is given pre-defined search terms (best-case for grep).
- 10 queries spanning architecture, proxy, core, features, providers, reliability.
