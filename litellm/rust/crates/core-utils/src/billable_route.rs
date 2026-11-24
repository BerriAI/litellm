//! `classify_billable_request`: which requests count toward the gateway request totals
//! (`LiteLLM_DailyGatewayRequests`), and the route each one counts under. The proxy's
//! `BillableRequestMetricsMiddleware` and the gateway's request volume middleware both classify
//! through here, so the two sides of a gateway in front of Python cannot drift.
//!
//! The route is taken from a closed list rather than the raw path, so the tables stay bounded.
//! Billing is a deliberate lower bound on SpendLogs rows: when this list and the proxy's routes
//! disagree, a request goes uncounted, never counted twice.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, strum::IntoStaticStr)]
pub enum Category {
    #[strum(serialize = "llm")]
    Llm,
    #[strum(serialize = "mcp")]
    Mcp,
    #[strum(serialize = "a2a")]
    A2a,
}

/// The POST inference surface that writes a SpendLogs row on success, matched by suffix. A longer
/// suffix comes before a shorter one it ends with, e.g. `/chat/completions` before `/completions`.
const LLM_ROUTE_SUFFIXES: [&str; 20] = [
    "/chat/completions",
    "/completions",
    "/embeddings",
    "/responses",
    "/rerank",
    "/moderations",
    "/images/generations",
    "/images/edits",
    "/images/variations",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/videos",
    // `/v1/videos/{id}/remix`.
    "/remix",
    "/ocr",
    // `/v1/search` and `/v1/vector_stores/{id}/search`.
    "/search",
    "/rag/query",
    "/rag/ingest",
    // Gemini-native `/v1beta/models/{model}:generateContent`.
    ":generateContent",
    ":streamGenerateContent",
];

/// Exact paths only: a suffix match would also catch resources that share the ending, e.g. the
/// OpenAI Assistants route `/v1/threads/{id}/messages`, which writes no SpendLogs row.
const LLM_ROUTE_EXACT: [&str; 6] = [
    "/v1/messages",
    // Google Interactions create; `/{id}` reads and `/cancel` do not match.
    "/interactions",
    "/v1beta/interactions",
    // AWS-SDK-shaped pass-through: the operation rides in the `X-Amz-Target` header.
    "/comprehendmedical",
    "/transcribe",
    "/eu_jev/v1/systemone",
];

/// Counts only an index search, `/opensearch/{index}/_search`, never document writes or lookups.
const OPENSEARCH: &str = "/opensearch";

/// `LiteLLMRoutes.mapped_pass_through_routes` without `/langfuse`, which forwards observability
/// traffic rather than inference, and `/opensearch`, which counts only an index search. A path
/// below one of these counts under the prefix. The proxy's tests check this list against the enum.
pub const PASSTHROUGH_PREFIXES: [&str; 31] = [
    "/backend-api/codex",
    "/bedrock",
    "/comprehendmedical",
    "/azure_speech",
    "/transcribe",
    "/vertex-ai",
    "/vertex_ai",
    "/cohere",
    "/cursor",
    "/cursor-cli",
    "/gemini",
    "/anthropic",
    "/azure",
    "/azure_ai",
    "/openai",
    "/openai_passthrough",
    "/assemblyai",
    "/eu.assemblyai",
    "/tinyfish",
    "/vllm",
    "/mistral",
    "/typesafe",
    "/laya",
    "/bespoke",
    "/openrouter",
    "/milvus",
    "/gigachat",
    "/watsonx",
    "/nvidia_nim",
    "/deepgram",
    "/fal_ai",
];

const MCP_MANAGEMENT_PREFIX: &str = "/v1/mcp";
/// The REST wrapper's tool call executes a tool and logs like the `/mcp` transport; its list and
/// test siblings do not count.
const MCP_REST_TOOL_CALL: &str = "/mcp-rest/tools/call";
/// Only `message/send` on a path counts: the bare `/a2a/{agent_id}` route carries its JSON-RPC
/// method in the body, where task RPCs that write no SpendLogs row ride too.
const A2A_INVOKE_SUFFIX: &str = "/message/send";
const A2A_TRANSPORT_PREFIXES: [&str; 2] = ["/v1/a2a/", "/a2a/"];

/// `path` past `prefix` and the slash after it.
fn below<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    path.strip_prefix(prefix)?.strip_prefix('/')
}

fn is_segment(text: &str) -> bool {
    !text.is_empty() && !text.contains('/')
}

/// `/(?:toolset/)?[^/]+/mcp` matched whole.
fn is_dynamic_mcp_transport(path: &str) -> bool {
    path.strip_prefix('/')
        .and_then(|path| path.strip_suffix("/mcp"))
        .is_some_and(|server| {
            is_segment(server) || server.strip_prefix("toolset/").is_some_and(is_segment)
        })
}

fn mcp_route(path: &str) -> Option<&'static str> {
    if path == MCP_MANAGEMENT_PREFIX || below(path, MCP_MANAGEMENT_PREFIX).is_some() {
        return None;
    }
    (path == "/mcp"
        || below(path, "/mcp").is_some()
        || path == MCP_REST_TOOL_CALL
        || is_dynamic_mcp_transport(path))
    .then_some("/mcp")
}

fn a2a_route(path: &str) -> Option<&'static str> {
    (path.ends_with(A2A_INVOKE_SUFFIX)
        && A2A_TRANSPORT_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix)))
    .then_some("/a2a")
}

fn llm_route(path: &str) -> Option<&'static str> {
    if let Some(rest) = below(path, OPENSEARCH) {
        let index_search = rest.strip_suffix("/_search").is_some_and(is_segment);
        return index_search.then_some(OPENSEARCH);
    }
    LLM_ROUTE_EXACT
        .into_iter()
        .find(|route| path == *route)
        .or_else(|| {
            LLM_ROUTE_SUFFIXES
                .into_iter()
                .find(|suffix| path.ends_with(suffix))
        })
        .or_else(|| {
            PASSTHROUGH_PREFIXES
                .into_iter()
                .find(|prefix| below(path, prefix).is_some())
        })
}

/// The category and the route a request to `path` with `method` counts under, or `None` when it
/// does not count. MCP and A2A count on any method, an inference route only on `POST`: non-POST
/// calls that log (pass-through reads) never count, so any drift is an undercount.
pub fn classify(path: &str, method: &str) -> Option<(Category, &'static str)> {
    let path = match path.trim_end_matches('/') {
        "" => "/",
        trimmed => trimmed,
    };
    if let Some(route) = mcp_route(path) {
        return Some((Category::Mcp, route));
    }
    if let Some(route) = a2a_route(path) {
        return Some((Category::A2a, route));
    }
    if !method.eq_ignore_ascii_case("POST") {
        return None;
    }
    llm_route(path).map(|route| (Category::Llm, route))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::messages("/v1/messages", Some((Category::Llm, "/v1/messages")))]
    #[case::messages_trailing_slash("/v1/messages/", Some((Category::Llm, "/v1/messages")))]
    #[case::ocr("/v1/ocr", Some((Category::Llm, "/ocr")))]
    #[case::bare_ocr("/ocr", Some((Category::Llm, "/ocr")))]
    #[case::chat_before_completions("/v1/chat/completions", Some((Category::Llm, "/chat/completions")))]
    #[case::completions("/v1/completions", Some((Category::Llm, "/completions")))]
    #[case::gemini_native(
        "/v1beta/models/gemini-pro:generateContent",
        Some((Category::Llm, ":generateContent"))
    )]
    #[case::vector_store_search("/v1/vector_stores/vs_1/search", Some((Category::Llm, "/search")))]
    #[case::assistants_messages_do_not_count("/v1/threads/thread_1/messages", None)]
    #[case::interactions("/v1beta/interactions", Some((Category::Llm, "/v1beta/interactions")))]
    #[case::interaction_read_does_not_count("/v1beta/interactions/abc", None)]
    #[case::comprehend_medical("/comprehendmedical", Some((Category::Llm, "/comprehendmedical")))]
    #[case::eu_jev_decision("/eu_jev/v1/systemone", Some((Category::Llm, "/eu_jev/v1/systemone")))]
    #[case::eu_jev_below_decision("/eu_jev/v1/systemone/extra", None)]
    #[case::bedrock_passthrough(
        "/bedrock/model/claude/invoke",
        Some((Category::Llm, "/bedrock"))
    )]
    #[case::bare_passthrough_prefix("/bedrock", None)]
    #[case::langfuse_does_not_count("/langfuse/api/public/ingestion", None)]
    #[case::opensearch_index_search("/opensearch/logs/_search", Some((Category::Llm, "/opensearch")))]
    #[case::opensearch_document_write("/opensearch/logs/_doc", None)]
    #[case::opensearch_deep_search("/opensearch/logs/extra/_search", None)]
    #[case::management("/key/generate", None)]
    #[case::root("/", None)]
    fn a_post_counts_only_on_an_inference_route(
        #[case] path: &str,
        #[case] expected: Option<(Category, &'static str)>,
    ) {
        assert_eq!(classify(path, "POST"), expected);
    }

    #[rstest]
    #[case::mcp("/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_below("/mcp/tools/list", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_rest_tool_call("/mcp-rest/tools/call", Some((Category::Mcp, "/mcp")))]
    #[case::mcp_rest_tool_list("/mcp-rest/tools/list", None)]
    #[case::dynamic_server("/github/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::toolset("/toolset/dev/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::server_named_toolset("/toolset/mcp", Some((Category::Mcp, "/mcp")))]
    #[case::too_deep("/a/b/c/mcp", None)]
    #[case::management("/v1/mcp/server", None)]
    #[case::a2a("/a2a/agent-1/message/send", Some((Category::A2a, "/a2a")))]
    #[case::a2a_v1("/v1/a2a/agent-1/message/send", Some((Category::A2a, "/a2a")))]
    #[case::bare_a2a("/a2a/agent-1", None)]
    #[case::inference_on_get("/v1/messages", None)]
    #[case::eu_jev_on_get("/eu_jev/v1/systemone", None)]
    fn mcp_and_a2a_count_on_any_method(
        #[case] path: &str,
        #[case] expected: Option<(Category, &'static str)>,
    ) {
        assert_eq!(classify(path, "GET"), expected);
    }

    #[rstest]
    fn the_method_is_matched_case_insensitively() {
        assert_eq!(
            classify("/v1/messages", "post"),
            Some((Category::Llm, "/v1/messages"))
        );
    }
}
