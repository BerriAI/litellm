# Vector Duplicate Detection — Arquitectura de Producción

## Contexto

El MVP actual (`feat/vector-duplicate-detection`) funciona como script CLI con un índice JSON y búsqueda lineal. Este documento describe los componentes necesarios para convertirlo en un servicio productivo.

---

## Componentes necesarios

### 1. Vector store (reemplaza el JSON)

El JSON funciona para el MVP pero no escala — la búsqueda es O(n) lineal por cada query. En producción se necesita un vector store con índice ANN (Approximate Nearest Neighbor):

| Opción | Cuándo usarla |
|---|---|
| **Qdrant** | Self-hosted, liviano, ideal para este caso — `docker pull qdrant/qdrant` y listo |
| **pgvector** | Si ya tienen PostgreSQL (ej. el Prisma DB del proxy de LiteLLM) — cero infra nueva |
| **Chroma** | Más simple que Qdrant, bueno para escala moderada |
| **Pinecone / Weaviate** | Si quieren managed sin ops, mayor escala |

Para BerriAI/litellm (~22k issues + PRs totales) Qdrant o pgvector son más que suficientes.

---

### 2. Indexador incremental (reemplaza el rebuild semanal)

El rebuild completo del índice tarda 50–80 min con Ollama local (ver [MVP notes](./vector-duplicate-detection-mvp.md)). En producción se indexa solo lo nuevo:

```
GitHub webhook (issue/PR closed o merged)
    → event processor
    → embed el item nuevo
    → upsert al vector store (keyed por número de issue/PR)
```

El rebuild completo pasa a ser una operación de bootstrapping única al iniciar el servicio.

---

### 3. Query service (API HTTP)

Un servicio FastAPI pequeño que expone tres endpoints:

```
POST /query
  body: { title, body, type: "issue" | "pr" }
  → embed on-the-fly
  → busca en vector store
  → retorna top-K matches con scores

POST /index
  body: { number, title, body, type, state }
  → embed
  → upsert al vector store

POST /scan
  → itera todos los open issues/PRs
  → query cada uno contra el índice
  → retorna resultados en JSON o CSV
```

Este servicio usa LiteLLM para embeddings — dogfooding del producto.

---

### 4. GitHub integration

Dos mecanismos posibles, no excluyentes:

**GitHub App (recomendado):**
- Recibe webhooks de `issues.opened`, `pull_request.opened`, `issues.closed`, `pull_request.merged`
- En `opened`: llama al query service; si score ≥ threshold, postea un comentario con el link al duplicado
- En `closed` / `merged`: llama al index endpoint para agregar el item al índice

**GitHub Actions (más simple, mayor latencia):**
- Workflow disparado en `on: [issues, pull_request]`
- Llama al query service vía HTTP desde el runner
- Postea el comentario con `gh` CLI o la GitHub API directamente

---

### 5. Modelo de embeddings

| Opción | Latencia | Costo | Privacidad |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | ~100ms | $0.02 / 1M tokens | datos salen |
| `voyage-code-3` (Voyage) | ~150ms | similar | datos salen |
| `nomic-embed-text` via Ollama (CPU) | ~500ms | $0 | todo local |
| `nomic-embed-text` via Ollama (GPU) | ~50ms | costo de infra | todo local |

Para producción en CI / GitHub Actions: **`text-embedding-3-small`** es el balance óptimo entre latencia y costo. Para entornos self-hosted o con requisitos de privacidad: Ollama en instancia con GPU.

El modelo se configura vía `LITELLM_EMBEDDING_MODEL` — sin cambios de código.

---

## Arquitectura mínima viable en producción

```
GitHub webhook
      │
      ▼
┌──────────────────────┐
│   GitHub App /       │   Cloud Run · Fly.io · EC2 nano
│   FastAPI service    │
│                      │
│   LiteLLM SDK        │──── Embedding API (OpenAI / Ollama)
│   (embeddings)       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Qdrant          │   mismo container o servicio separado
│   vector store       │
└──────────────────────┘
```

### Estimación de costos para BerriAI/litellm

| Ítem | Estimación |
|---|---|
| ~500 nuevos issues/PRs por semana | ~500 embeddings/semana |
| Costo de embeddings (OpenAI) | < $0.01 / semana |
| Servicio API (Cloud Run, escala a cero) | ~$5–10 / mes |
| Qdrant (mismo container o instancia small) | $0–5 / mes |
| **Total estimado** | **< $15 / mes** |

---

## Estado actual vs. lo que falta

| Componente | Estado | Ubicación |
|---|---|---|
| Lógica de embedding + cosine search | ✅ implementado | `.github/scripts/vector_duplicate_detection.py` |
| Index builder (bulk) | ✅ implementado | comando `index` |
| Query por item | ✅ implementado | comando `query` |
| Scan batch | ✅ implementado | comando `scan` |
| Vector store persistente | ❌ solo JSON en disco | — |
| Indexado incremental via webhook | ❌ no implementado | — |
| API HTTP (FastAPI) | ❌ no implementado | — |
| GitHub App / webhook consumer | ❌ no implementado | — |
| Comentario automático en GitHub | ❌ lógica existe, no conectada al webhook | `post_comment()` en el script |

El core algorítmico está completo. Lo que falta es el **wrapper de infraestructura** alrededor de la lógica existente.

---

## Resultados del MVP (2026-03-07)

Runs realizados contra BerriAI/litellm con índice de 2,000 items y `nomic-embed-text` via Ollama:

| Run | Índice | Queried | Flagged | Threshold | CSV |
|---|---|---|---|---|---|
| Issues | 2,000 closed issues | 100 open issues | 8 | 0.87 | `docs/vdd-mvp-issues-2k.csv` |
| PRs | 2,000 merged PRs | 100 open PRs | 17 | 0.87 | `docs/vdd-mvp-prs-2k.csv` |

Destacados:

- `#22758` ↔ `#22759` — **duplicado exacto** (Docker Compose, score 0.9972)
- `#22183` → `#18261` — mismo bug `disable_exception_on_block`, reportado dos veces con 4 meses de diferencia (score 0.8828)
- `#23030` → `#22769` — `chore: regenerate poetry.lock` idéntico, PR posiblemente stale (score 0.9952)
- `#22979` → `#22933` — mismo fix `sanitize empty text content blocks`, worth checking if already merged (score 0.9613)

Ver [MVP notes](./vector-duplicate-detection-mvp.md) para instrucciones de ejecución y análisis del bottleneck de indexado.
