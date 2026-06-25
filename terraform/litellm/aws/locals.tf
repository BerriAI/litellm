# Gateway path prefixes — mirrored verbatim from gateway/routes/allowlist.py
# and the helm ingress in helm/litellm/templates/ingress.yaml. Anything not in
# this list and not a UI asset path falls through to the backend (management
# API) catch-all rule on the ALB.
#
# ALB listener rules cap path-pattern conditions at 5 values per rule, so we
# chunk this list and emit one rule per chunk.
locals {
  # Every resource the stack creates is named `<tenant>-litellm-<env>`
  # (or that with a per-resource suffix). Computed once here so the rest of
  # the stack can reference local.name.
  name = "${var.tenant}-litellm-${var.env}"

  # This is a reusable module — it declares no `provider` block, so the AWS
  # provider's `default_tags` is the caller's concern, not ours. To keep the
  # same per-resource tagging the stack had when it owned the provider, the
  # module threads `local.tags` onto every taggable resource itself. Callers
  # may layer org-wide tags on top via their own provider `default_tags`
  # (those merge with these). `var.tags` is the per-deployment override.
  tags = merge(
    {
      "litellm:stack" = local.name
      "managed-by"    = "terraform"
    },
    var.tags,
  )

  gateway_path_prefixes = [
    "/v1/chat/*", "/chat/*",
    "/v1/completions*", "/completions*",
    "/v1/embeddings*", "/embeddings*",
    "/v1/moderations*", "/moderations*",
    "/v1/audio/*", "/audio/*",
    "/v1/images/*", "/images/*",
    "/v1/files*", "/files*",
    "/v1/batches*", "/batches*",
    "/v1/fine_tuning/*", "/fine_tuning/*",
    "/v1/fine-tuning/*", "/fine-tuning/*",
    "/v1/responses*", "/responses*",
    "/v1/threads*", "/threads*",
    "/v1/assistants*", "/assistants*",
    "/v1/vector_stores*", "/vector_stores*",
    "/v1/indexes*",
    "/v1/models*", "/models*",
    "/openai/*", "/engines/*",
    "/v1/messages*", "/messages*",
    "/v1/skills/*", "/v1/a2a/*",
    "/v1/rerank*", "/v2/rerank*", "/rerank*",
    "/v1/ocr*", "/ocr*",
    "/v1/rag/*", "/rag/*",
    "/v1/video/*", "/v1/videos/*", "/video/*", "/videos/*",
    "/v1/search*", "/search*",
    "/v1/containers/*", "/containers/*",
    "/v1/evals/*",
    "/v1/memory/*",
    "/queue/chat/*",
    "/v1beta/*",
    "/interactions/*",
    "/anthropic/*", "/azure/*", "/azure_ai/*", "/aws/*", "/bedrock/*",
    "/cohere/*", "/gemini/*", "/google/*",
    "/vertex_ai/*", "/vertex-ai/*",
    "/assemblyai/*", "/eu.assemblyai/*",
    "/langfuse/*", "/vllm/*",
    "/mistral/*", "/groq/*", "/voyage/*", "/cursor/*", "/milvus/*",
    "/openai_passthrough/*",
    "/toolset/*",
    "/v1/realtime*", "/realtime*",
    "/health*", "/metrics", "/test*",
  ]

  # Static UI asset prefixes — handled by the UI service, not the backend
  # catch-all. /favicon.ico and / are also UI but added as exact rules.
  ui_path_prefixes = [
    "/litellm-asset-prefix/*",
    "/_next/*",
    "/assets/*",
    "/ui/*",
  ]

  ui_exact_paths = [
    "/",
    "/favicon.ico",
    "/ui",
  ]

  # ALB rules accept ≤ 5 path-pattern values per condition. Chunk the prefix
  # list so each chunk becomes one rule.
  gateway_path_chunks = chunklist(local.gateway_path_prefixes, 5)
}
