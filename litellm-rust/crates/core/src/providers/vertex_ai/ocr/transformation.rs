use crate::constants::{
    BEARER_SCHEME, GOOGLE_API_KEY_PREFIX, VERTEX_GLOBAL_API_BASE, VERTEX_GLOBAL_LOCATION,
};
use crate::error::{json_type_name, CoreError, CoreResult};
use crate::ocr::transformation::{OcrAuth, OcrProviderConfig};
use crate::ocr::types::{OcrRequestData, OcrResponseData};
use serde_json::{json, Map, Value};

use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

const VERTEX_DEFAULT_LOCATION: &str = "us-central1";
const VERTEX_DEFAULT_DEEPSEEK_API_BASE: &str = VERTEX_GLOBAL_API_BASE;
const VERTEX_AI_API_KEY_ENV: &str = "VERTEX_AI_API_KEY";
const VERTEXAI_API_KEY_ENV: &str = "VERTEXAI_API_KEY";
const VERTEXAI_PROJECT_ENV: &str = "VERTEXAI_PROJECT";
const VERTEXAI_LOCATION_ENV: &str = "VERTEXAI_LOCATION";
const VERTEX_LOCATION_ENV: &str = "VERTEX_LOCATION";

#[rustfmt::skip]
const DEEPSEEK_SUPPORTED_OCR_PARAMS: &[&str] = &[
    "stream",
    "temperature",
    "max_tokens",
    "top_p",
    "n",
    "stop",
];

pub struct VertexAiOcrConfig;
pub struct VertexAiDeepSeekOcrConfig;

pub const VERTEX_AI_OCR_CONFIG: VertexAiOcrConfig = VertexAiOcrConfig;
pub const VERTEX_AI_DEEPSEEK_OCR_CONFIG: VertexAiDeepSeekOcrConfig = VertexAiDeepSeekOcrConfig;

fn string_param<'a>(params: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| params.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

pub fn is_deepseek_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("deepseek")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertexTokenSource {
    Explicit(String),
    Mint,
}

fn is_google_api_key(token: &str) -> bool {
    token.starts_with(GOOGLE_API_KEY_PREFIX)
}

fn google_api_key_not_oauth_error() -> CoreError {
    CoreError::Auth(
        "Received a Google API key (AIza...) for Vertex AI, which is not an OAuth access token. \
         Provide service-account credentials/ADC or an OAuth access token instead"
            .to_string(),
    )
}

pub fn classify_vertex_bearer(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<VertexTokenSource> {
    let token = api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(VERTEX_AI_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .or_else(|| env_lookup(VERTEXAI_API_KEY_ENV).filter(|key| !key.trim().is_empty()));

    match token {
        Some(token) if is_google_api_key(token.trim()) => Err(google_api_key_not_oauth_error()),
        Some(token) => Ok(VertexTokenSource::Explicit(token.trim().to_string())),
        None => Ok(VertexTokenSource::Mint),
    }
}

fn malformed_vertex_authorization_error() -> CoreError {
    CoreError::Auth(
        "Vertex AI requires exactly one `Authorization: Bearer <OAuth access token>` header. \
         Provide a valid OAuth Bearer token, or omit the header to mint one from credentials/ADC"
            .to_string(),
    )
}

pub fn validate_vertex_authorization_headers(values: &[&str]) -> CoreResult<()> {
    match values {
        [] => Ok(()),
        [single] => validate_vertex_authorization_value(single),
        _ => Err(malformed_vertex_authorization_error()),
    }
}

fn validate_vertex_authorization_value(header_value: &str) -> CoreResult<()> {
    let mut parts = header_value.split_whitespace();
    let scheme = parts
        .next()
        .ok_or_else(malformed_vertex_authorization_error)?;
    let token = parts
        .next()
        .ok_or_else(malformed_vertex_authorization_error)?;
    if parts.next().is_some() {
        return Err(malformed_vertex_authorization_error());
    }
    if !scheme.eq_ignore_ascii_case(BEARER_SCHEME) {
        return Err(malformed_vertex_authorization_error());
    }
    if is_google_api_key(token) {
        return Err(google_api_key_not_oauth_error());
    }
    Ok(())
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

fn vertex_base_url(location: &str) -> String {
    match location {
        VERTEX_GLOBAL_LOCATION => VERTEX_GLOBAL_API_BASE.to_string(),
        location if !location.contains('-') => {
            format!("https://aiplatform.{location}.rep.googleapis.com")
        }
        location => format!("https://{location}-aiplatform.googleapis.com"),
    }
}

fn vertex_mistral_api_base(api_base: Option<&str>, location: &str) -> String {
    api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| vertex_base_url(location))
        .trim_end_matches('/')
        .to_string()
}

pub fn complete_vertex_mistral_url(
    api_base: Option<&str>,
    model: &str,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let project = vertex_project(optional_params, env_lookup)?;
    let location = vertex_location(optional_params, env_lookup);
    let base = vertex_mistral_api_base(api_base, &location);
    Ok(format!(
        "{base}/v1/projects/{project}/locations/{location}/publishers/mistralai/models/{model}:rawPredict"
    ))
}

pub fn complete_vertex_deepseek_url(
    api_base: Option<&str>,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let project = vertex_project(optional_params, env_lookup)?;
    let location = vertex_location(optional_params, env_lookup);
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(VERTEX_DEFAULT_DEEPSEEK_API_BASE)
        .trim_end_matches('/');
    Ok(format!(
        "{base}/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
    ))
}

fn document_content_item(document: &Value) -> CoreResult<Value> {
    let object = document.as_object().ok_or_else(|| CoreError::InvalidType {
        expected: "object",
        actual: json_type_name(document),
    })?;
    let doc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("document.type"))?;
    let url_field = match doc_type {
        "image_url" => "image_url",
        "document_url" => "document_url",
        other => {
            return Err(CoreError::InvalidRequest(format!(
                "Unsupported document type: {other}. Expected 'image_url' or 'document_url'"
            )))
        }
    };
    let url = object
        .get(url_field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(CoreError::MissingField(url_field))?;

    Ok(json!({
        "type": "image_url",
        "image_url": url,
    }))
}

fn deepseek_model_name(model: &str) -> String {
    if model.starts_with("deepseek-ai/") {
        model.to_string()
    } else {
        format!("deepseek-ai/{model}")
    }
}

fn first_choice_content(response: &Value) -> CoreResult<Value> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .cloned()
        .filter(|content| match content {
            Value::String(value) => !value.is_empty(),
            Value::Object(_) => true,
            _ => false,
        })
        .ok_or_else(|| {
            CoreError::InvalidResponse("No content in DeepSeek OCR response".to_string())
        })
}

fn ocr_data_from_content(content: Value, usage: Option<Value>, model: &str) -> Value {
    match content {
        Value::String(content) => {
            if content.trim_start().starts_with('{') {
                serde_json::from_str(&content).unwrap_or_else(|_| {
                    json!({
                        "pages": [{"index": 0, "markdown": content}],
                        "model": model,
                        "usage_info": usage.unwrap_or_else(|| json!({})),
                    })
                })
            } else {
                json!({
                    "pages": [{"index": 0, "markdown": content}],
                    "model": model,
                    "usage_info": usage.unwrap_or_else(|| json!({})),
                })
            }
        }
        Value::Object(_) => content,
        other => json!({
            "pages": [{"index": 0, "markdown": other.to_string()}],
            "model": model,
            "usage_info": usage.unwrap_or_else(|| json!({})),
        }),
    }
}

impl OcrProviderConfig for VertexAiOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        MISTRAL_OCR_CONFIG.supported_ocr_params()
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        MISTRAL_OCR_CONFIG.transform_ocr_request(model, document, optional_params)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        MISTRAL_OCR_CONFIG.transform_ocr_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_vertex_mistral_url(api_base, model, optional_params, env_lookup)
    }

    fn ocr_auth(&self) -> OcrAuth {
        OcrAuth::VertexOauth
    }

    fn requires_data_uri_document(&self) -> bool {
        true
    }
}

impl OcrProviderConfig for VertexAiDeepSeekOcrConfig {
    fn supported_ocr_params(&self) -> &'static [&'static str] {
        DEEPSEEK_SUPPORTED_OCR_PARAMS
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> CoreResult<OcrRequestData> {
        let mut data = Map::new();
        data.insert(
            "model".to_string(),
            Value::String(deepseek_model_name(model)),
        );
        data.insert(
            "messages".to_string(),
            json!([{"role": "user", "content": [document_content_item(&document)?]}]),
        );
        for (key, value) in optional_params {
            if DEEPSEEK_SUPPORTED_OCR_PARAMS.contains(&key.as_str()) {
                data.insert(key, value);
            }
        }
        Ok(OcrRequestData {
            data: Value::Object(data),
            files: None,
        })
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<OcrResponseData> {
        let response = response_json
            .as_object()
            .ok_or_else(|| CoreError::unexpected_response_type(&response_json))?;
        let usage = response.get("usage").cloned();
        let content = first_choice_content(&response_json)?;
        let mut ocr_data = ocr_data_from_content(content.clone(), usage.clone(), model);

        if !ocr_data.get("pages").is_some_and(Value::is_array) {
            ocr_data = json!({
                "pages": [{
                    "index": 0,
                    "markdown": match content {
                        Value::String(value) => value,
                        other => other.to_string(),
                    }
                }],
                "model": ocr_data.get("model").and_then(Value::as_str).unwrap_or(model),
                "usage_info": ocr_data.get("usage_info").cloned().or(usage).unwrap_or_else(|| json!({})),
            });
        }

        let object = ocr_data
            .as_object()
            .ok_or_else(|| CoreError::unexpected_response_type(&ocr_data))?;
        let pages = object
            .get("pages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let usage_info = object
            .get("usage_info")
            .cloned()
            .or_else(|| response.get("usage").cloned());
        Ok(OcrResponseData {
            pages,
            model: object
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(model)
                .to_string(),
            document_annotation: object.get("document_annotation").cloned(),
            usage_info,
            object: "ocr".to_string(),
        })
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_vertex_deepseek_url(api_base, optional_params, env_lookup)
    }

    fn ocr_auth(&self) -> OcrAuth {
        OcrAuth::VertexOauth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_mistral_url_uses_project_location_and_model() {
        let params = Map::from_iter([
            ("vertex_project".to_string(), json!("proj-1")),
            ("vertex_location".to_string(), json!("europe-west4")),
        ]);

        let url = complete_vertex_mistral_url(None, "mistral-ocr-maas", &params, &|_| None)
            .expect("url builds");

        assert_eq!(
            url,
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }

    #[test]
    fn vertex_mistral_url_uses_global_host_without_region_prefix() {
        let params = Map::from_iter([
            ("vertex_project".to_string(), json!("proj-1")),
            ("vertex_location".to_string(), json!("global")),
        ]);

        let url = complete_vertex_mistral_url(None, "mistral-ocr-maas", &params, &|_| None)
            .expect("url builds");

        assert_eq!(
            url,
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }

    #[test]
    fn vertex_mistral_url_uses_residency_host_for_single_token_location() {
        let params = Map::from_iter([
            ("vertex_project".to_string(), json!("proj-1")),
            ("vertex_location".to_string(), json!("eu")),
        ]);

        let url = complete_vertex_mistral_url(None, "mistral-ocr-maas", &params, &|_| None)
            .expect("url builds");

        assert_eq!(
            url,
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/proj-1/locations/eu/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }

    #[test]
    fn vertex_mistral_url_prefers_explicit_api_base() {
        let params = Map::from_iter([
            ("vertex_project".to_string(), json!("proj-1")),
            ("vertex_location".to_string(), json!("global")),
        ]);

        let url = complete_vertex_mistral_url(
            Some("https://custom.example.com/"),
            "mistral-ocr-maas",
            &params,
            &|_| None,
        )
        .expect("url builds");

        assert_eq!(
            url,
            "https://custom.example.com/v1/projects/proj-1/locations/global/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
    }

    #[test]
    fn classify_vertex_bearer_uses_explicit_oauth_token() {
        let source = classify_vertex_bearer(Some("  ya29.oauth-token  "), &|_| None)
            .expect("token classifies");
        assert_eq!(
            source,
            VertexTokenSource::Explicit("ya29.oauth-token".to_string())
        );
    }

    #[test]
    fn classify_vertex_bearer_reads_oauth_token_from_env() {
        let source = classify_vertex_bearer(None, &|key| {
            (key == VERTEX_AI_API_KEY_ENV).then(|| "ya29.from-env".to_string())
        })
        .expect("token classifies");
        assert_eq!(
            source,
            VertexTokenSource::Explicit("ya29.from-env".to_string())
        );
    }

    #[test]
    fn classify_vertex_bearer_mints_when_no_token_supplied() {
        let source = classify_vertex_bearer(None, &|_| None).expect("token classifies");
        assert_eq!(source, VertexTokenSource::Mint);
    }

    #[test]
    fn classify_vertex_bearer_rejects_google_api_key() {
        let err = classify_vertex_bearer(Some("AIzaSyExampleApiKeyValue"), &|_| None)
            .expect_err("google api key is rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn classify_vertex_bearer_rejects_google_api_key_from_env() {
        let err = classify_vertex_bearer(None, &|key| {
            (key == VERTEXAI_API_KEY_ENV).then(|| "AIzaSyExampleApiKeyValue".to_string())
        })
        .expect_err("google api key from env is rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn validate_vertex_authorization_headers_rejects_api_key_bearer() {
        for header in [
            "Bearer AIzaSyExampleApiKeyValue",
            "  bearer   AIzaSyExampleApiKeyValue  ",
            "BEARER AIzaSyExampleApiKeyValue",
        ] {
            let err = validate_vertex_authorization_headers(&[header])
                .expect_err(&format!("api key bearer rejected: {header:?}"));
            assert!(matches!(err, CoreError::Auth(_)), "{header:?} -> {err:?}");
        }
    }

    #[test]
    fn validate_vertex_authorization_headers_rejects_malformed_values() {
        for header in [
            "",
            "   ",
            "AIzaSyExampleApiKeyValue",
            "ya29.raw-token-without-scheme",
            "Basic dXNlcjpwYXNz",
            "Token ya29.some-token",
            "Bearer2 ya29.token",
            "Bearer",
            "Bearer    ",
            "Bearer ya29.token extra-part",
            "Bearer ya29.token AIzaExtra",
        ] {
            let err = validate_vertex_authorization_headers(&[header])
                .expect_err(&format!("expected rejection for {header:?}"));
            assert!(matches!(err, CoreError::Auth(_)), "{header:?} -> {err:?}");
        }
    }

    #[test]
    fn validate_vertex_authorization_headers_rejects_duplicate_headers() {
        let err = validate_vertex_authorization_headers(&[
            "Bearer ya29.first-token",
            "Bearer ya29.second-token",
        ])
        .expect_err("duplicate authorization headers rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn validate_vertex_authorization_headers_allows_single_oauth_bearer() {
        for header in [
            "Bearer ya29.real-oauth-token",
            "bearer ya29.real-oauth-token",
            "BEARER ya29.real-oauth-token",
            "  Bearer   ya29.real-oauth-token  ",
        ] {
            validate_vertex_authorization_headers(&[header])
                .unwrap_or_else(|err| panic!("oauth bearer allowed: {header:?} -> {err:?}"));
        }
        validate_vertex_authorization_headers(&[])
            .expect("no authorization header defers to minting");
    }

    #[test]
    fn vertex_configs_use_google_oauth() {
        assert_eq!(VERTEX_AI_OCR_CONFIG.ocr_auth(), OcrAuth::VertexOauth);
        assert_eq!(
            VERTEX_AI_DEEPSEEK_OCR_CONFIG.ocr_auth(),
            OcrAuth::VertexOauth
        );
    }

    #[test]
    fn vertex_mistral_reuses_mistral_body_transform() {
        let body = VERTEX_AI_OCR_CONFIG
            .transform_ocr_request(
                "mistral-ocr-maas",
                json!({"type": "image_url", "image_url": "data:image/png;base64,abc"}),
                Map::new(),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "mistral-ocr-maas");
        assert_eq!(body["document"]["image_url"], "data:image/png;base64,abc");
    }

    #[test]
    fn vertex_deepseek_request_uses_ocr_endpoint_shape() {
        let body = VERTEX_AI_DEEPSEEK_OCR_CONFIG
            .transform_ocr_request(
                "deepseek-ocr-maas",
                json!({"type": "document_url", "document_url": "gs://bucket/doc.pdf"}),
                Map::from_iter([("temperature".to_string(), json!(0.1))]),
            )
            .expect("request transforms")
            .data;

        assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(body["temperature"], 0.1);
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({"type": "image_url", "image_url": "gs://bucket/doc.pdf"})
        );
    }

    #[test]
    fn vertex_deepseek_response_wraps_markdown_content() {
        let response = VERTEX_AI_DEEPSEEK_OCR_CONFIG
            .transform_ocr_response(
                "deepseek-ocr-maas",
                json!({
                    "choices": [{"message": {"content": "# OCR text"}}],
                    "usage": {"prompt_tokens": 1}
                }),
            )
            .expect("response transforms");

        assert_eq!(
            response.pages,
            vec![json!({"index": 0, "markdown": "# OCR text"})]
        );
        assert_eq!(response.model, "deepseek-ocr-maas");
        assert_eq!(response.usage_info, Some(json!({"prompt_tokens": 1})));
    }
}
