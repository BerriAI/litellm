use super::RouteSet;

/// The `LiteLLMRoutes` groups the core's own rules read. The host sends every group, and the rest
/// are reached only by name from an operator's `allowed_routes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub enum RouteGroup {
    #[strum(serialize = "public_routes")]
    Public,
    #[strum(serialize = "master_key_only_routes")]
    MasterKeyOnly,
    #[strum(serialize = "mcp_inference_routes")]
    McpInference,
    #[strum(serialize = "agent_inference_routes")]
    AgentInference,
}

pub(crate) const MANAGED_DISCOVERY_ROUTES: &[&str] = &["/agents", "/v1/agents"];
pub(crate) const MANAGED_REALTIME_ROUTES: &[&str] = &[
    "/realtime",
    "/v1/realtime",
    "/openai/v1/realtime",
    "/realtime/translations",
    "/v1/realtime/translations",
];
pub(crate) const MANAGED_LIVE_ROUTES: &[&str] = &[
    "/live/sessions",
    "/v1/live/sessions",
    "/openai/v1/live/sessions",
];
pub(crate) const MANAGED_MODEL_PREFIXES: &[&str] = &["", "/v1"];
pub(crate) const MANAGED_MODEL_OPERATIONS: &[&str] = &[
    "chat/completions",
    "completions",
    "embeddings",
    "responses",
    "messages",
    "messages/count_tokens",
    "images/generations",
    "images/edits",
    "audio/transcriptions",
    "audio/speech",
    "moderations",
    "rerank",
    "decisions",
    "systemone",
    "extraction",
    "ocr",
];
pub(crate) const MANAGED_MODEL_EXTRA_ROUTES: &[&str] = &[
    "/openai/v1/responses",
    "/v2/rerank",
    "/claude_code_gateway/v1/messages",
    "/claude_code_gateway/v1/messages/count_tokens",
    "/cursor/chat/completions",
];
pub(crate) static MANAGED_MODEL_PATHS: RouteSet = RouteSet::new(&[
    "/engines/{model:path}/chat/completions",
    "/engines/{model:path}/completions",
    "/engines/{model:path}/embeddings",
    "/openai/deployments/{model:path}/chat/completions",
    "/openai/deployments/{model:path}/completions",
    "/openai/deployments/{model:path}/embeddings",
    "/openai/deployments/{model:path}/images/generations",
    "/openai/deployments/{model:path}/images/edits",
    "/v1beta/models/{model_name:path}:countTokens",
    "/v1beta/models/{model_name:path}:generateContent",
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    "/models/{model_name:path}:countTokens",
    "/models/{model_name:path}:generateContent",
    "/models/{model_name:path}:streamGenerateContent",
]);
/// MCP inference routes a managed agent may call; the OAuth token endpoints are not among them.
pub(crate) const MANAGED_MCP_EXCLUDED: &[&str] = &["/token", "/introspect"];
