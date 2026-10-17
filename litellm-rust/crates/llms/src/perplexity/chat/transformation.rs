//! `litellm/llms/perplexity/chat/transformation.py`: Perplexity through the OpenAI SDK at
//! `https://api.perplexity.ai`, with the subset of OpenAI params it takes, LiteLLM's
//! `X-Pplx-Integration` attribution header, and the citations of an answer added to its
//! message as `url_citation` annotations with the search counts in the usage.

use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use serde_json::{Map, Value, json};

use crate::openai::chat::compatible::{
    CompatibleContext, OpenAiCompatibleChatConfig, ResponseShape, citation_markers,
    sdk_complete_url,
};

const SUPPORTED_PARAMS: &[&str] = &[
    "frequency_penalty",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "response_format",
    "stream",
    "temperature",
    "top_p",
    "max_retries",
    "extra_headers",
];
const ATTRIBUTION_HEADER: (&str, &str) = ("X-Pplx-Integration", "litellm");

pub struct PerplexityChatConfig;

pub const PERPLEXITY_CHAT_CONFIG: PerplexityChatConfig = PerplexityChatConfig;

impl OpenAiCompatibleChatConfig for PerplexityChatConfig {
    fn provider(&self) -> LlmProviders {
        LlmProviders::Perplexity
    }

    fn label(&self) -> &'static str {
        "Perplexity"
    }

    fn api_base_env(&self) -> &'static [&'static str] {
        &["PERPLEXITY_API_BASE"]
    }

    fn default_api_base(&self) -> Option<&'static str> {
        Some("https://api.perplexity.ai")
    }

    fn api_key_env(&self) -> &'static [&'static str] {
        &["PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"]
    }

    fn complete_url(&self, api_base: &str) -> String {
        sdk_complete_url(api_base)
    }

    fn headers(
        &self,
        _body: &Map<String, Value>,
        _lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Vec<(String, String)> {
        vec![(ATTRIBUTION_HEADER.0.into(), ATTRIBUTION_HEADER.1.into())]
    }

    fn supported_params(&self, context: &CompatibleContext<'_>) -> Vec<&'static str> {
        let mut params = SUPPORTED_PARAMS.to_vec();
        if context.supports(self.name(), "supports_reasoning") {
            params.push("reasoning_effort");
        }
        if context.supports(self.name(), "supports_web_search") {
            params.push("web_search_options");
        }
        params
    }

    /// `_enhance_usage_with_perplexity_fields` and `_add_citations_as_annotations`.
    fn fix_response(&self, body: &mut Map<String, Value>, _shape: &ResponseShape) -> bool {
        let usage_changed = enhance_usage(body);
        add_citations_as_annotations(body) || usage_changed
    }
}

/// The search count Perplexity reports, wherever it put it.
fn search_queries(body: &Map<String, Value>) -> Option<u64> {
    let usage = body.get("usage");
    let field = |value: Option<&Value>, key: &str| value.and_then(|v| v.get(key)).cloned();
    [
        field(usage, "num_search_queries"),
        body.get("num_search_queries").cloned(),
        field(usage, "search_queries"),
        body.get("search_queries").cloned(),
    ]
    .into_iter()
    .flatten()
    .find(|value| !value.is_null())
    .and_then(|value| value.as_u64())
}

fn enhance_usage(body: &mut Map<String, Value>) -> bool {
    let citation_chars: usize = match body.get("citations") {
        Some(Value::Array(citations)) => citations
            .iter()
            .filter(|citation| crate::base_llm::chat::params::truthy(citation))
            .map(|citation| match citation {
                Value::String(text) => text.chars().count(),
                other => other.to_string().chars().count(),
            })
            .sum(),
        _ => 0,
    };
    let citation_tokens = match citation_chars {
        0 => 0,
        chars => (chars / 4).max(1),
    };
    let searches = search_queries(body).filter(|count| *count > 0);
    if citation_tokens == 0 && searches.is_none() {
        return false;
    }
    let usage = body
        .entry("usage")
        .or_insert_with(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}));
    if usage.is_null() {
        *usage = json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0});
    }
    let Some(usage) = usage.as_object_mut() else {
        return false;
    };
    if citation_tokens > 0 {
        usage.insert("citation_tokens".into(), json!(citation_tokens));
    }
    if let Some(searches) = searches {
        let details = usage
            .entry("prompt_tokens_details")
            .or_insert_with(|| json!({}));
        if !details.is_object() {
            *details = json!({});
        }
        if let Some(details) = details.as_object_mut() {
            details.insert("web_search_requests".into(), json!(searches));
        }
    }
    true
}

fn add_citations_as_annotations(body: &mut Map<String, Value>) -> bool {
    let citations: Vec<Option<String>> = match body.get("citations") {
        Some(Value::Array(citations)) => citations
            .iter()
            .map(|citation| citation.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    };
    let titles: Vec<(String, String)> = match body.get("search_results") {
        Some(Value::Array(results)) => results
            .iter()
            .filter_map(|result| {
                Some((
                    result.get("url")?.as_str()?.to_owned(),
                    result.get("title")?.as_str()?.to_owned(),
                ))
            })
            .collect(),
        _ => Vec::new(),
    };
    let Some(message) = body
        .get_mut("choices")
        .and_then(Value::as_array_mut)
        .and_then(|choices| choices.first_mut())
        .and_then(|choice| choice.get_mut("message"))
        .and_then(Value::as_object_mut)
    else {
        return false;
    };
    let Some(content) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
    else {
        return false;
    };
    let annotations: Vec<Value> = citation_markers(content)
        .into_iter()
        .filter_map(|(number, start, end)| {
            let url = citations.get(number.checked_sub(1)?)?.as_ref()?;
            let title = titles
                .iter()
                .rev()
                .find(|(known, _)| known == url)
                .map_or("", |(_, title)| title.as_str());
            Some(json!({"type": "url_citation", "url_citation": {
                "url": url, "title": title, "start_index": start, "end_index": end,
            }}))
        })
        .collect();
    if annotations.is_empty() {
        return false;
    }
    match message.get_mut("annotations") {
        Some(Value::Array(existing)) => existing.extend(annotations),
        _ => {
            message.insert("annotations".into(), Value::Array(annotations));
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        Error, ErrorDetail,
        openai::chat::compatible::{fix_response, transform_request},
    };

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("an object")
        };
        map
    }

    fn send(params: Value) -> Result<Map<String, Value>, Error> {
        transform_request(
            &PERPLEXITY_CHAT_CONFIG,
            &CompatibleContext {
                model: "sonar",
                drop_params: false,
                model_info: None,
            },
            object(params),
        )
        .map(|sent| sent.body)
    }

    #[rstest]
    fn max_completion_tokens_goes_as_sent() {
        let sent = send(
            json!({"messages": [], "max_completion_tokens": 7, "search_domain_filter": ["a.com"]}),
        )
        .unwrap();
        assert_eq!(
            Value::Object(sent),
            json!({"model": "sonar", "messages": [], "max_completion_tokens": 7, "search_domain_filter": ["a.com"]})
        );
    }

    #[rstest]
    #[case::tools(json!({"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}]}))]
    #[case::seed(json!({"messages": [], "seed": 1}))]
    fn params_perplexity_lacks_are_refused(#[case] params: Value) {
        assert!(matches!(
            send(params),
            Err(Error::InvalidRequest(ErrorDetail::UnsupportedParams { .. }))
        ));
    }

    #[rstest]
    fn citations_become_annotations_and_searches_reach_the_usage() {
        let mut body = object(json!({
            "model": "sonar",
            "citations": ["https://a.example", "https://b.example"],
            "search_results": [{"url": "https://b.example", "title": "B"}],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "é see [2] and [3]"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "num_search_queries": 2},
        }));
        assert!(fix_response(
            &PERPLEXITY_CHAT_CONFIG,
            &mut body,
            &ResponseShape::default()
        ));
        assert_eq!(
            body["choices"][0]["message"]["annotations"],
            json!([{"type": "url_citation", "url_citation": {"url": "https://b.example", "title": "B", "start_index": 6, "end_index": 9}}])
        );
        assert_eq!(body["usage"]["citation_tokens"], json!(8));
        assert_eq!(
            body["usage"]["prompt_tokens_details"],
            json!({"web_search_requests": 2})
        );
    }
}
