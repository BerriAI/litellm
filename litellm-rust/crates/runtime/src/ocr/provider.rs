use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderTransformation;
use litellm_core::providers::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use litellm_core::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use litellm_core::providers::vertex_ai::ocr::transformation::{
    VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG,
};
use serde_json::{Map, Value};

use super::types::{OcrAuthStrategy, OcrResponseHandling, OcrRuntimeConfig};
use crate::constants::{
    AZURE_AI_API_BASE_ENV, AZURE_AI_API_KEY_ENV, AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV,
    AZURE_DOCUMENT_INTELLIGENCE_API_VERSION, AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
    MISTRAL_API_KEY_ENV, MISTRAL_DEFAULT_API_BASE, VERTEX_AI_API_KEY_ENV,
    VERTEX_DEFAULT_DEEPSEEK_API_BASE, VERTEX_DEFAULT_LOCATION, VERTEX_LOCATION_ENV,
    VERTEXAI_API_KEY_ENV, VERTEXAI_LOCATION_ENV, VERTEXAI_PROJECT_ENV,
};

struct MistralRuntimeConfig;
struct AzureAiRuntimeConfig;
struct AzureDocumentIntelligenceRuntimeConfig;
struct VertexAiRuntimeConfig;
struct VertexAiDeepSeekRuntimeConfig;

static MISTRAL_RUNTIME_CONFIG: MistralRuntimeConfig = MistralRuntimeConfig;
static AZURE_AI_RUNTIME_CONFIG: AzureAiRuntimeConfig = AzureAiRuntimeConfig;
static AZURE_DOCUMENT_INTELLIGENCE_RUNTIME_CONFIG: AzureDocumentIntelligenceRuntimeConfig =
    AzureDocumentIntelligenceRuntimeConfig;
static VERTEX_AI_RUNTIME_CONFIG: VertexAiRuntimeConfig = VertexAiRuntimeConfig;
static VERTEX_AI_DEEPSEEK_RUNTIME_CONFIG: VertexAiDeepSeekRuntimeConfig =
    VertexAiDeepSeekRuntimeConfig;

pub(super) fn ocr_provider_config(
    provider: &str,
    model: &str,
) -> Option<&'static dyn OcrRuntimeConfig> {
    match provider {
        "mistral" => Some(&MISTRAL_RUNTIME_CONFIG),
        "azure_ai" if is_azure_document_intelligence_model(model) => {
            Some(&AZURE_DOCUMENT_INTELLIGENCE_RUNTIME_CONFIG)
        }
        "azure_ai" => Some(&AZURE_AI_RUNTIME_CONFIG),
        "vertex_ai" if model.to_ascii_lowercase().contains("deepseek") => {
            Some(&VERTEX_AI_DEEPSEEK_RUNTIME_CONFIG)
        }
        "vertex_ai" => Some(&VERTEX_AI_RUNTIME_CONFIG),
        _ => None,
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn resolve_value(
    explicit: Option<&str>,
    env_name: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    missing_message: &str,
) -> CoreResult<String> {
    non_empty(explicit)
        .map(str::to_string)
        .or_else(|| env_lookup(env_name).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| CoreError::Auth(missing_message.to_string()))
}

fn resolve_vertex_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(VERTEX_AI_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .or_else(|| env_lookup(VERTEXAI_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing Vertex AI access token - pass api_key or provide Authorization via extra_headers"
                    .to_string(),
            )
        })
}

fn string_param<'a>(params: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| params.get(*key).and_then(Value::as_str))
        .and_then(|value| non_empty(Some(value)))
}

fn vertex_project(
    params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    string_param(params, &["vertex_project", "vertex_ai_project"])
        .map(str::to_string)
        .or_else(|| env_lookup(VERTEXAI_PROJECT_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::InvalidRequest(
                "Missing vertex_project - Set VERTEXAI_PROJECT environment variable or pass vertex_project parameter"
                    .to_string(),
            )
        })
}

fn vertex_location(
    params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    string_param(params, &["vertex_location", "vertex_ai_location"])
        .map(str::to_string)
        .or_else(|| env_lookup(VERTEXAI_LOCATION_ENV).filter(|value| !value.trim().is_empty()))
        .or_else(|| env_lookup(VERTEX_LOCATION_ENV).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| VERTEX_DEFAULT_LOCATION.to_string())
}

fn encode_model_id(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![byte as char]
            }
            _ => format!("%{byte:02X}").chars().collect(),
        })
        .collect()
}

fn pages_token_is_valid(token: &str) -> bool {
    let mut parts = token.split('-');
    let Some(start) = parts.next() else {
        return false;
    };
    if start.is_empty() || !start.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty() && end.chars().all(|ch| ch.is_ascii_digit()) && parts.next().is_none()
        }
    }
}

fn normalize_pages_param(pages: &Value) -> CoreResult<Option<String>> {
    match pages {
        Value::String(value) => {
            let normalized = value
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(",");
            if normalized.split(',').all(pages_token_is_valid) {
                Ok(Some(normalized))
            } else {
                Err(CoreError::InvalidRequest(format!(
                    "Invalid `pages` string for Azure Document Intelligence: {value:?}. Expected format like '1-3,5,7-9'."
                )))
            }
        }
        Value::Array(values) if values.is_empty() => Ok(None),
        Value::Array(values) if values.iter().all(Value::is_i64) => {
            let pages = values
                .iter()
                .map(|value| value.as_i64().expect("checked is_i64"))
                .map(|page| {
                    (page >= 0).then_some(page + 1).ok_or_else(|| {
                        CoreError::InvalidRequest(
                            "`pages` integers must be >= 0 (Mistral 0-based indices)".to_string(),
                        )
                    })
                })
                .collect::<CoreResult<std::collections::BTreeSet<_>>>()?;
            Ok(Some(
                pages
                    .into_iter()
                    .map(|page| page.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            ))
        }
        Value::Array(values) if values.iter().all(Value::is_string) => {
            let normalized = values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(",");
            if normalized.split(',').all(pages_token_is_valid) {
                Ok(Some(normalized))
            } else {
                Err(CoreError::InvalidRequest(format!(
                    "Invalid `pages` list for Azure Document Intelligence: {values:?}. Expected tokens like '1' or '3-5'."
                )))
            }
        }
        _ => Err(CoreError::InvalidRequest(
            "`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'."
                .to_string(),
        )),
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

impl OcrRuntimeConfig for MistralRuntimeConfig {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation {
        &MISTRAL_OCR_CONFIG
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let base = non_empty(api_base)
            .unwrap_or(MISTRAL_DEFAULT_API_BASE)
            .trim_end_matches('/');
        Ok(if base.ends_with("/v1") {
            format!("{base}/ocr")
        } else {
            format!("{base}/v1/ocr")
        })
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        non_empty(api_key)
            .map(str::to_string)
            .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
            .ok_or_else(|| CoreError::Auth("Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params".to_string()))
    }
}

impl OcrRuntimeConfig for AzureAiRuntimeConfig {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation {
        &AZURE_AI_OCR_CONFIG
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let base = resolve_value(
            api_base,
            AZURE_AI_API_BASE_ENV,
            env_lookup,
            "Missing Azure AI API Base - Set AZURE_AI_API_BASE environment variable or pass api_base parameter",
        )?;
        Ok(format!(
            "{}/providers/mistral/azure/ocr",
            base.trim_end_matches('/')
        ))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_value(
            api_key,
            AZURE_AI_API_KEY_ENV,
            env_lookup,
            "Missing Azure AI API Key - A call is being made to Azure AI but no key is set either in the environment variables or via params",
        )
    }

    fn requires_data_uri_document(&self) -> bool {
        true
    }
}

impl OcrRuntimeConfig for AzureDocumentIntelligenceRuntimeConfig {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation {
        &AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let endpoint = resolve_value(
            api_base,
            AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT_ENV,
            env_lookup,
            "Missing Azure Document Intelligence Endpoint - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT environment variable or pass api_base parameter",
        )?;
        let mut url = format!(
            "{}/documentintelligence/documentModels/{}:analyze?api-version={}",
            endpoint.trim_end_matches('/'),
            encode_model_id(model),
            AZURE_DOCUMENT_INTELLIGENCE_API_VERSION
        );
        if let Some(pages) = optional_params.get("pages")
            && let Some(normalized) = normalize_pages_param(pages)?
        {
            url.push_str("&pages=");
            url.push_str(&normalized);
        }
        Ok(url)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_value(
            api_key,
            AZURE_DOCUMENT_INTELLIGENCE_API_KEY_ENV,
            env_lookup,
            "Missing Azure Document Intelligence API Key - Set AZURE_DOCUMENT_INTELLIGENCE_API_KEY environment variable or pass api_key parameter",
        )
    }

    fn auth_strategy(&self) -> OcrAuthStrategy {
        OcrAuthStrategy::Header("Ocp-Apim-Subscription-Key")
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::AzureDocumentIntelligencePoll
    }
}

impl OcrRuntimeConfig for VertexAiRuntimeConfig {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation {
        &VERTEX_AI_OCR_CONFIG
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let project = vertex_project(optional_params, env_lookup)?;
        let location = vertex_location(optional_params, env_lookup);
        let base = non_empty(api_base)
            .map(str::to_string)
            .unwrap_or_else(|| format!("https://{location}-aiplatform.googleapis.com"));
        Ok(format!(
            "{}/v1/projects/{project}/locations/{location}/publishers/mistralai/models/{model}:rawPredict",
            base.trim_end_matches('/')
        ))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_vertex_api_key(api_key, env_lookup)
    }

    fn requires_data_uri_document(&self) -> bool {
        true
    }
}

impl OcrRuntimeConfig for VertexAiDeepSeekRuntimeConfig {
    fn transformation(&self) -> &'static dyn OcrProviderTransformation {
        &VERTEX_AI_DEEPSEEK_OCR_CONFIG
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let project = vertex_project(optional_params, env_lookup)?;
        let location = vertex_location(optional_params, env_lookup);
        let base = non_empty(api_base).unwrap_or(VERTEX_DEFAULT_DEEPSEEK_API_BASE);
        Ok(format!(
            "{}/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions",
            base.trim_end_matches('/')
        ))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_vertex_api_key(api_key, env_lookup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_supports_ocr_providers() {
        assert!(ocr_provider_config("mistral", "mistral-ocr-latest").is_some());
        assert!(
            ocr_provider_config("azure_ai", "pixtral-12b-2409")
                .expect("azure ai config resolves")
                .requires_data_uri_document()
        );
        assert_eq!(
            ocr_provider_config("azure_ai", "doc-intelligence/prebuilt-read")
                .expect("document intelligence config resolves")
                .response_handling(),
            OcrResponseHandling::AzureDocumentIntelligencePoll
        );
        assert!(
            ocr_provider_config("vertex_ai", "deepseek-ocr-maas")
                .expect("vertex deepseek config resolves")
                .transformation()
                .get_supported_ocr_params()
                .contains(&"temperature")
        );
        assert!(ocr_provider_config("openai", "gpt-4o").is_none());
    }

    #[test]
    fn mistral_url_and_credentials_resolve_in_runtime() {
        assert_eq!(
            MISTRAL_RUNTIME_CONFIG
                .complete_url(None, "mistral-ocr-latest", &Map::new(), &|_| None)
                .expect("url builds"),
            "https://api.mistral.ai/v1/ocr"
        );
        assert_eq!(
            MISTRAL_RUNTIME_CONFIG
                .complete_url(
                    Some("https://proxy.internal/v1/"),
                    "mistral-ocr-latest",
                    &Map::new(),
                    &|_| None,
                )
                .expect("url builds"),
            "https://proxy.internal/v1/ocr"
        );
        let env_lookup = |key: &str| (key == MISTRAL_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(
            MISTRAL_RUNTIME_CONFIG
                .resolve_api_key(Some("  "), &env_lookup)
                .expect("env key resolves"),
            "sk-env"
        );
    }

    #[test]
    fn document_intelligence_url_normalizes_zero_based_pages() {
        let params = Map::from_iter([("pages".to_string(), serde_json::json!([2, 0, 2]))]);
        let url = AZURE_DOCUMENT_INTELLIGENCE_RUNTIME_CONFIG
            .complete_url(
                Some("https://example.cognitiveservices.azure.com/"),
                "azure_ai/doc-intelligence/prebuilt-layout",
                &params,
                &|_| None,
            )
            .expect("url builds");
        assert_eq!(
            url,
            "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&pages=1,3"
        );
    }

    #[test]
    fn vertex_url_uses_project_location_and_model() {
        let params = Map::from_iter([
            ("vertex_project".to_string(), serde_json::json!("proj-1")),
            (
                "vertex_location".to_string(),
                serde_json::json!("europe-west4"),
            ),
        ]);
        let url = VERTEX_AI_RUNTIME_CONFIG
            .complete_url(None, "mistral-ocr-maas", &params, &|_| None)
            .expect("url builds");
        assert_eq!(
            url,
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }
}
