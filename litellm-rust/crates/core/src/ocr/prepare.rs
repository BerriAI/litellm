use serde::Serialize;
use serde_json::Value;

use super::OcrClient;
use super::hooks::OcrDuringCallRequest;
use super::types::{LiteLLMOcrRequest, OcrConnection, OcrDocument, PreparedOcrRequest};

pub(crate) async fn transform_request_body<B>(
    client: &OcrClient,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    retains_document: bool,
    body: B,
    validate: impl Fn(&Value) -> Result<(), super::Error>,
) -> Result<reqwest::Request, super::Error>
where
    B: Serialize,
{
    let composed = crate::call_arguments::compose_body(
        &request.optional_params,
        &body,
        request.config.get_supported_ocr_params(&request.model),
    )?;
    validate(&composed)?;
    let (body, headers) = if request.hooks.intercepts_requests() {
        let body = composed;
        let retained_fields = request
            .optional_params
            .keys()
            .filter(|name| body.get(*name).is_some())
            .cloned()
            .chain(retains_document.then(|| "document".to_string()))
            .filter(|name| {
                request
                    .optional_params
                    .get("extra_body")
                    .and_then(Value::as_object)
                    .is_none_or(|overrides| !overrides.contains_key(name))
            })
            .collect();
        let changed = request
            .hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: request.provider_name().into(),
                url: url.into(),
                headers: headers.to_vec(),
                body,
                retained_fields,
            })
            .await?;
        if !changed.body.is_object() {
            return Err(super::Error::RequestField {
                path: "guardrail.body".into(),
            });
        }
        validate(&changed.body)?;
        (changed.body, changed.headers)
    } else {
        (composed, headers.to_vec())
    };
    build_http_request(client, request, url, &headers, &body)
}

pub(crate) fn build_http_request<B: Serialize>(
    client: &OcrClient,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: &B,
) -> Result<reqwest::Request, super::Error> {
    let builder = client
        .provider_http()
        .post(url)
        .json(body)
        .timeout(request.connection.timeout);
    crate::http_utils::with_headers(builder, headers, crate::http_utils::HeaderPolicy::All)
        .build()
        .map_err(crate::transport::Error::from)
        .map_err(super::Error::from)
}

pub(crate) async fn guardrail_document(
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
) -> Result<(OcrDocument, Vec<(String, String)>), super::Error> {
    if !request.hooks.intercepts_requests() {
        return Ok((request.document.clone(), headers.to_vec()));
    }
    let changed = request
        .hooks
        .during_call(OcrDuringCallRequest {
            model: request.model.clone(),
            custom_llm_provider: request.provider_name().into(),
            url: url.into(),
            headers: headers.to_vec(),
            body: serde_json::to_value(&request.document).map_err(|_| {
                super::Error::RequestField {
                    path: "document".into(),
                }
            })?,
            retained_fields: Vec::new(),
        })
        .await?;
    let document = super::json::decode_request_value(changed.body, "guardrail.document")?;
    Ok((document, changed.headers))
}

pub(crate) fn body_document(body: &Value) -> Result<OcrDocument, super::Error> {
    let document = body
        .get("document")
        .and_then(Value::as_object)
        .ok_or_else(|| super::Error::RequestField {
            path: "body.document".into(),
        })?;
    let source = document
        .iter()
        .filter(|(name, _)| matches!(name.as_str(), "type" | "image_url" | "document_url"))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    super::json::decode_request_value(Value::Object(source), "body.document")
}

pub(crate) fn credential_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

pub(crate) fn prepare_request(request: LiteLLMOcrRequest) -> PreparedOcrRequest {
    use litellm_auth::{InputSource, Sourced};

    let credentials = request.credentials.clone();
    let api_base_env = match request.config.provider() {
        super::provider_config::OcrProvider::Mistral => Some("MISTRAL_API_BASE"),
        super::provider_config::OcrProvider::AzureAi => Some("AZURE_AI_API_BASE"),
        super::provider_config::OcrProvider::Cohere
        | super::provider_config::OcrProvider::Reducto
        | super::provider_config::OcrProvider::VertexAi => None,
    };
    let dynamic_api_key = credentials.dynamic_api_key.or_else(|| {
        credentials.api_key.clone().or_else(|| {
            request
                .config
                .get_api_key_env_var()
                .and_then(credential_env)
                .map(|value| Sourced::new(value, InputSource::Environment))
        })
    });
    let dynamic_api_base = credentials.dynamic_api_base.or_else(|| {
        credentials.api_base.clone().or_else(|| {
            api_base_env
                .and_then(credential_env)
                .map(|value| Sourced::new(value, InputSource::Environment))
        })
    });
    let resolved = request
        .config
        .resolve_connection_params(super::types::OcrCredentialInputs {
            dynamic_api_key,
            dynamic_api_base,
            ..credentials
        });
    let transport = request.transport.clone();
    PreparedOcrRequest::new(request, OcrConnection::new(resolved, transport))
}
