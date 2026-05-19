# Gateway path prefixes — mirrored verbatim from gateway/routes/allowlist.py
# and helm/litellm/templates/ingress.yaml. URL maps use the "path matcher"
# rule with `paths` lists; up to 10 path globs per rule, up to 50 rules
# per matcher. Easily fits the gateway list in one rule per chunk-of-10.
locals {
  # Every resource the stack creates is named `${tenant}-litellm-${env}`
  # (or that with a per-resource suffix). Computed once here so the rest of
  # the stack can reference local.name.
  name = "${var.tenant}-litellm-${var.env}"

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

  ui_path_prefixes = [
    "/",
    "/favicon.ico",
    "/litellm-asset-prefix/*",
    "/_next/*",
    "/assets/*",
    "/ui",
    "/ui/*",
  ]

  proxy_config_enabled = length(keys(var.proxy_config)) > 0
  proxy_config_b64     = local.proxy_config_enabled ? base64encode(yamlencode(var.proxy_config)) : ""

  proxy_config_env = local.proxy_config_enabled ? [
    { name = "LITELLM_PROXY_CONFIG_B64", value = local.proxy_config_b64 },
    { name = "CONFIG_FILE_PATH", value = "/tmp/litellm-config.yaml" },
  ] : []

  # Resolved image URIs: per-component override wins, otherwise compose
  # from image_registry + image_tag. Cloud Run only accepts AR / gcr.io /
  # docker.io paths — see variables.tf for the full constraint list.
  gateway_image    = var.gateway_image != "" ? var.gateway_image : "${var.image_registry}/litellm-gateway:${var.image_tag}"
  backend_image    = var.backend_image != "" ? var.backend_image : "${var.image_registry}/litellm-backend:${var.image_tag}"
  ui_image         = var.ui_image != "" ? var.ui_image : "${var.image_registry}/litellm-ui:${var.image_tag}"
  migrations_image = var.migrations_image != "" ? var.migrations_image : "${var.image_registry}/litellm-migrations:${var.image_tag}"
}
