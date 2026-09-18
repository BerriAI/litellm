use serde::Serialize;
use serde_json::Value;

use super::OcrClient;
use super::hooks::OcrDuringCallRequest;
use super::types::{OcrConnection, OcrDocument, PreparedOcrRequest, ResolvedOcrRequest};

pub(crate) async fn transform_request_body<B>(
    client: &OcrClient,
    request: &PreparedOcrRequest,
    url: &str,
    headers: &[(String, String)],
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
    let retained_fields = request
        .optional_params
        .keys()
        .filter(|name| composed.get(*name).is_some())
        .cloned()
        .chain(retains_document(&composed, &request.document).then(|| "document".to_string()))
        .collect();
    let (body, headers) = if request.hooks.intercepts_requests() {
        let changed = request
            .hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: request.provider_name().into(),
                api_key: request.connection.api_key.clone(),
                url: url.into(),
                headers: headers.to_vec(),
                body: composed,
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
            api_key: request.connection.api_key.clone(),
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

fn retains_document(composed: &Value, document: &OcrDocument) -> bool {
    body_document(composed).is_ok_and(|sent| sent.source() == document.source())
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

pub(crate) fn prepare_request(request: ResolvedOcrRequest) -> PreparedOcrRequest {
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::call_arguments::{CallArguments, compose_body, parse_options};

    #[derive(serde::Deserialize)]
    struct KnownParams {
        pages: Option<Vec<i64>>,
    }

    #[test]
    fn parsed_provider_params_separates_known_and_extra_params() {
        let arguments: CallArguments = serde_json::from_value(json!({
            "pages": [0, 2],
            "future_ocr_option": true,
            "extra_body": {"provider_option": "value"}
        }))
        .unwrap();
        let known: KnownParams = parse_options(&arguments).unwrap();
        assert_eq!(known.pages, Some(vec![0, 2]));
        assert_eq!(arguments["future_ocr_option"], true);
        assert_eq!(arguments["extra_body"], json!({"provider_option": "value"}));
        assert_eq!(
            arguments
                .iter()
                .filter(|(name, _)| name.as_str() != "pages")
                .count(),
            2
        );
        assert_eq!(
            compose_body(&arguments, &json!({"pages": known.pages}), &["pages"]).unwrap(),
            json!({
                "pages": [0, 2], "future_ocr_option": true, "provider_option": "value"
            })
        );
    }
}
