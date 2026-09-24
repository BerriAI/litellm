# Per-component path prefixes mirrored verbatim from the AWS module's
# gateway_path_prefixes / ui_path_prefixes blocks (and ultimately from
# gateway/routes/allowlist.py plus the helm ingress in
# helm/litellm/templates/ingress.yaml). Anything not in either list and not
# a UI asset path falls through to the backend (management API) on the
# Application Gateway URL path map.
#
# Application Gateway URL path map rules cap path-based conditions per rule
# differently per SKU; for the path-based backend pool strategy used here
# we keep a single combined list and emit one rule per prefix.

locals {
  # Every Azure resource the stack creates is named `<tenant>-litellm-<env>`
  # (or that with a per-resource suffix). Computed once so the rest of the
  # stack can reference `local.name`.
  name = "${var.tenant}-litellm-${var.env}"

  # Module-level tagging. Caller-provided provider default_tags merge with
  # these at apply time.
  tags = merge(
    {
      "litellm:stack" = local.name
      "managed-by"    = "terraform"
    },
    var.tags,
  )

  # Resource group that owns every resource the module creates. Caller may
  # supply an existing one with var.resource_group_name; otherwise the
  # module creates one named `<tenant>-litellm-<env>-rg`.
  resource_group_name = var.resource_group_name != "" ? var.resource_group_name : "${local.name}-rg"

  # Gateway data-plane path prefixes (mirrors AWS gateway_path_prefixes).
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

  # Static UI asset prefixes (handled by the ui Container App, not the
  # backend catch-all). / and /favicon.ico are added as exact-match paths.
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

  # TLS is enabled when a Key Vault certificate ID is supplied AND
  # plaintext is disallowed (the default).
  tls_enabled = var.key_vault_certificate_id != "" && !var.allow_plaintext_app_gateway

  # Postgres admin setup: the first active-directory admin that provisions
  # the Flexible Server. We use the current principal invoking terraform.
  # (The `azure_ad_admin` block on the PG server accepts a specific object
  # ID; callers needing a different admin can pass it through
  # var.db_ad_admin_object_id.)
  db_ad_admin_object_id_default = try(data.azurerm_client_config.current.object_id, "")

  # Image URI resolution (mirrors AWS / GCP defaults):
  #   <image_registry>/<image_namespace>/litellm-<component>:<image_tag>
  gateway_image_resolved    = coalesce(var.gateway_image, "${var.image_registry}/${var.image_namespace}/litellm-gateway:${var.image_tag}")
  backend_image_resolved    = coalesce(var.backend_image, "${var.image_registry}/${var.image_namespace}/litellm-backend:${var.image_tag}")
  ui_image_resolved         = coalesce(var.ui_image, "${var.image_registry}/${var.image_namespace}/litellm-ui:${var.image_tag}")
  migrations_image_resolved = coalesce(var.migrations_image, "${var.image_registry}/${var.image_namespace}/litellm-migrations:${var.image_tag}")

  # Proxy config as YAML, uploaded to the storage account blob
  # `config/litellm-config.yaml`. The Container Apps download this on
  # startup via azure-identity and set `CONFIG_FILE_PATH=/tmp/litellm-config.yaml`.
  proxy_config_yaml = var.proxy_config != {} ? yamlencode(var.proxy_config) : ""
}
