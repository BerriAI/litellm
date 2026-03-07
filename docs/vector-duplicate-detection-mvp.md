# Vector Duplicate Detection — MVP Notes

Branch: `feat/vector-duplicate-detection`

## What it does

Replaces keyword-based duplicate detection with embedding-powered semantic search. New issues and PRs are embedded and compared against a vector index of historical closed/merged items. If similarity exceeds a threshold, a comment is posted linking to the likely duplicate.

**Components:**
- `.github/scripts/vector_duplicate_detection.py` — core logic (index, query, scan)
- `.github/workflows/vector_duplicate_index.yml` — weekly scheduled index rebuild
- `.github/workflows/vector_duplicate_check.yml` — per-issue/PR trigger

**Embedding backend:** Any model supported by LiteLLM — OpenAI `text-embedding-3-small`, Voyage `voyage-code-3`, or local Ollama models. Configured via `LITELLM_EMBEDDING_MODEL`.

---

## Running a quick MVP test (Ollama + nomic-embed-text)

### Prerequisites

1. Ollama running locally with `nomic-embed-text` pulled:
   ```bash
   ollama pull nomic-embed-text
   ```

2. Python with LiteLLM installed:
   ```bash
   /opt/homebrew/bin/python3.13 -m pip install litellm typing_extensions --break-system-packages
   ```

3. `gh` CLI authenticated:
   ```bash
   gh auth status
   ```

### Step 1 — Verify embeddings work

```bash
/opt/homebrew/bin/python3.13 -c "
import litellm
r = litellm.embedding(model='ollama/nomic-embed-text', input=['test'])
print(f'OK — vector dim: {len(r.data[0][\"embedding\"])}')
"
```

Expected output: `OK — vector dim: 768`

### Step 2 — Build a small index (200 recent closed issues)

For a quick smoke test, limit to 200 issues to avoid the full-corpus bottleneck:

```bash
GITHUB_TOKEN=$(gh auth token) \
LITELLM_EMBEDDING_MODEL="ollama/nomic-embed-text" \
VECTOR_INDEX_PATH="/tmp/vdd-small-index.json" \
/opt/homebrew/bin/python3.13 -u -c "
import os, sys, json, math, subprocess
sys.path.insert(0, '.')
import litellm

REPO = 'BerriAI/litellm'
BATCH_SIZE = 20

def embed(texts):
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        b = texts[i:i+BATCH_SIZE]
        r = litellm.embedding(model='ollama/nomic-embed-text', input=b)
        out.extend([x['embedding'] for x in r.data])
        print(f'  embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}')
    return out

raw = subprocess.run(
    ['gh','issue','list','--repo',REPO,'--state','closed','--limit','200',
     '--json','number,title,body,labels,closedAt'],
    capture_output=True, text=True, check=True
).stdout
issues = json.loads(raw)
print(f'Fetched {len(issues)} closed issues. Building embeddings...')

texts = [f\"Issue #{i['number']}: {i['title']}\n\n{(i.get('body') or '')[:500]}\" for i in issues]
meta  = [{'type':'issue','number':i['number'],'title':i['title']} for i in issues]
vecs  = embed(texts)

index = [{'meta': m, 'vec': v} for m, v in zip(meta, vecs)]
with open('/tmp/vdd-small-index.json', 'w') as f:
    json.dump(index, f)
print(f'Index saved — {len(index)} entries.')
"
```

Takes ~2 minutes on a local machine (Ollama + M-series chip).

### Step 3 — Query the index

```bash
/opt/homebrew/bin/python3.13 -u -c "
import sys, json, math
sys.path.insert(0, '.')
import litellm

with open('/tmp/vdd-small-index.json') as f:
    index = json.load(f)

def embed_one(text):
    return litellm.embedding(model='ollama/nomic-embed-text', input=[text]).data[0]['embedding']

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na and nb else 0.0

def query(title, body='', threshold=0.78, top_k=3):
    vec = embed_one(f'{title}\n\n{body[:500]}')
    scores = sorted(
        [{'score': cosine(vec, e['vec']), **e['meta']} for e in index],
        key=lambda x: -x['score']
    )
    return [s for s in scores if s['score'] >= threshold][:top_k]

tests = [
    ('Thinking blocks corrupted on round-trip with multiple tool calls', ''),
    ('Add mercury 2 model to model_prices_and_context_window.json', ''),
    ('Bedrock tool use returns duplicate toolUse IDs', ''),
]

for title, body in tests:
    hits = query(title, body)
    print(f'\nQuery: \"{title[:70]}\"')
    for h in hits:
        print(f'  [{h[\"score\"]:.3f}] #{h[\"number\"]}: {h[\"title\"][:70]}')
    if not hits:
        print('  (no matches above threshold)')
"
```

### Observed results (2026-03-07)

| Open issue | Top match | Score |
|---|---|---|
| `[Bug]: Thinking blocks corrupted on round-trip…` | `[Bug]: Incorrect handling of ContentBlocks (Anthropic) for Code Execution` | 0.782 |
| `Add "mercury 2" model…` | `Add "opus-4-6" via Azure in model_prices_and_context_window.json` | 0.816 |
| `Add "mercury 2" model…` | `Add "mistral-large-2512" in model_prices_and_context_window.json` | 0.797 |

The model-addition matches (0.816, 0.797) are genuine duplicates — exactly the signal the system is designed to surface.

---

## Bottleneck: full-corpus indexing

The `index` command in the script fetches **three separate batches** via blocking `gh` calls before any embedding starts:

```
gh issue list --state closed --limit 5000   # up to 5,000 items
gh pr list    --state merged  --limit 5000   # up to 5,000 items
gh pr list    --state closed  --limit 5000   # up to 5,000 items
```

BerriAI/litellm has 22,000+ merged PRs. Each `gh` call pages through the API synchronously and buffers the full response before returning — no output is visible until it completes. On a full run this silent wait is **10–30 minutes** before the first embedding even starts.

**Contributing factors:**
- `subprocess.run(capture_output=True)` — no streaming, full buffering
- Python stdout fully buffered when writing to a file (vs. a tty) — progress prints are invisible until the buffer flushes
- Ollama is slower than a hosted API (~0.5s per batch of 20 vs. ~0.1s with OpenAI)

### Workarounds for local testing

Use `-u` (unbuffered Python) so prints appear immediately, and reduce the limit:

```bash
GITHUB_TOKEN=$(gh auth token) \
LITELLM_EMBEDDING_MODEL="ollama/nomic-embed-text" \
VECTOR_INDEX_PATH="/tmp/vdd-full.json" \
/opt/homebrew/bin/python3.13 -u \
  .github/scripts/vector_duplicate_detection.py index --repo BerriAI/litellm
```

> Note: even with `-u`, the first print won't appear until the first `gh` call returns (~5–10 min for 5,000 closed issues). This is expected.

---

## Running the real full index

Use this when you want a production-quality index for a batch scan:

```bash
GITHUB_TOKEN=$(gh auth token) \
LITELLM_EMBEDDING_MODEL="ollama/nomic-embed-text" \
VECTOR_INDEX_PATH="/tmp/vdd-full-index.json" \
SIMILARITY_THRESHOLD="0.82" \
/opt/homebrew/bin/python3.13 -u \
  .github/scripts/vector_duplicate_detection.py index --repo BerriAI/litellm
```

Expected timeline on local hardware (Ollama, M-series):

| Phase | Time estimate |
|---|---|
| Fetch 5,000 closed issues | ~5–10 min |
| Fetch 5,000 merged PRs | ~10–15 min |
| Fetch 5,000 closed PRs | ~10–15 min |
| Embed ~15,000 items @ 20/batch | ~25–40 min |
| **Total** | **~50–80 min** |

Once built, **query and scan are fast** (seconds per item). The index is saved as a JSON file and reused until the next weekly rebuild.

### Switching to a faster embedding model

For CI or recurring runs, use OpenAI (much faster, lower latency):

```bash
LITELLM_EMBEDDING_MODEL="text-embedding-3-small" \
OPENAI_API_KEY="sk-..." \
VECTOR_INDEX_PATH="/tmp/vdd-full-index.json" \
python3 -u .github/scripts/vector_duplicate_detection.py index --repo BerriAI/litellm
```

Full index with OpenAI takes ~5–10 min (API rate limits permitting).

### Running a batch scan after indexing

Once the index exists, scan all current open issues/PRs for duplicates:

```bash
GITHUB_TOKEN=$(gh auth token) \
LITELLM_EMBEDDING_MODEL="ollama/nomic-embed-text" \
VECTOR_INDEX_PATH="/tmp/vdd-full-index.json" \
/opt/homebrew/bin/python3.13 -u \
  .github/scripts/vector_duplicate_detection.py scan --repo BerriAI/litellm
```
