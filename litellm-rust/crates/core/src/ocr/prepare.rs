use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::OcrClient;
use super::error::{OcrError, OcrRequestError};
use super::hooks::OcrDuringCallRequest;
use super::types::{LiteLLMOcrRequest, OcrDocument};

pub(crate) use crate::params::ParsedProviderParams;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn _prepare_ocr_request<T: DeserializeOwned>(
    request: &LiteLLMOcrRequest,
) -> Result<ParsedProviderParams<T>, OcrRequestError> {
    crate::params::body_overrides(&request.optional_params).map_err(parameter_error)?;
    super::wire::decode_request_value(
        Value::Object(request.optional_params.clone()),
        "optional_params",
    )
}

pub(crate) fn merge_extra_params<B: Serialize>(
    body: &B,
    extra_params: crate::params::OpaqueFields,
) -> Result<Value, OcrRequestError> {
    crate::params::compose_body(body, &extra_params, &[]).map_err(parameter_error)
}

fn parameter_error(error: crate::params::Error) -> OcrRequestError {
    OcrRequestError::RequestField {
        path: match error {
            crate::params::Error::ExtraBody => "extra_body",
            crate::params::Error::Body => "body",
        }
        .into(),
    }
}

pub(crate) async fn transform_request_body<B>(
    client: &OcrClient,
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
    retains_document: bool,
    body: B,
    validate: impl FnOnce(&B) -> Result<(), OcrRequestError>,
) -> Result<reqwest::Request, OcrError>
where
    B: Serialize + DeserializeOwned,
{
    let consumed = super::wire::consumed_optional_param_names(
        &request.model,
        Some(request.adapter.provider().as_str()),
    )?;
    let body = crate::params::compose_body(&body, &request.optional_params, &consumed)
        .map_err(parameter_error)?;
    let (body, headers) = if request.hooks.intercepts_requests() {
        let overrides =
            crate::params::body_overrides(&request.optional_params).map_err(parameter_error)?;
        let retained_fields = request
            .optional_params
            .keys()
            .filter(|name| body.get(*name).is_some())
            .filter(|name| !overrides.is_some_and(|fields| fields.contains_key(*name)))
            .cloned()
            .chain(
                (retains_document
                    && !overrides.is_some_and(|fields| fields.contains_key("document")))
                .then(|| "document".to_string()),
            )
            .collect();
        let changed = request
            .hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: request.adapter.provider().as_str().into(),
                url: url.into(),
                headers: headers.to_vec(),
                body,
                retained_fields,
            })
            .await?;
        let projected: B =
            super::wire::decode_request_value(changed.body.clone(), "guardrail.body")?;
        validate(&projected)?;
        (changed.body, changed.headers)
    } else {
        let projected: B = super::wire::decode_request_value(body.clone(), "body")?;
        validate(&projected)?;
        (body, headers.to_vec())
    };
    build_http_request(client, request, url, &headers, &body)
}

pub(crate) fn build_http_request<B: Serialize>(
    client: &OcrClient,
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: &B,
) -> Result<reqwest::Request, OcrError> {
    let builder = client
        .provider_http()
        .post(url)
        .json(body)
        .timeout(request.connection.timeout);
    crate::http_utils::with_headers(builder, headers, crate::http_utils::HeaderPolicy::All)
        .build()
        .map_err(crate::error::TransportError::from)
        .map_err(OcrError::from)
}

pub(crate) async fn guardrail_document(
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
) -> Result<(OcrDocument, Vec<(String, String)>), OcrError> {
    if !request.hooks.intercepts_requests() {
        return Ok((request.document.clone(), headers.to_vec()));
    }
    let changed = request
        .hooks
        .during_call(OcrDuringCallRequest {
            model: request.model.clone(),
            custom_llm_provider: request.adapter.provider().as_str().into(),
            url: url.into(),
            headers: headers.to_vec(),
            body: serde_json::to_value(&request.document).map_err(|_| {
                OcrRequestError::RequestField {
                    path: "document".into(),
                }
            })?,
            retained_fields: Vec::new(),
        })
        .await?;
    let document = super::wire::decode_request_value(changed.body, "guardrail.document")?;
    Ok((document, changed.headers))
}

pub(crate) fn credential_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq)]
    struct KnownParams {
        pages: Option<Vec<i64>>,
    }

    #[test]
    fn parsed_provider_params_separates_known_and_extra_params() {
        let parsed: ParsedProviderParams<KnownParams> = super::super::wire::decode_request_value(
            json!({
                "pages": [0, 2],
                "future_ocr_option": true,
                "extra_body": {"provider_option": "value"}
            }),
            "optional_params",
        )
        .unwrap();

        assert_eq!(parsed.known.pages, Some(vec![0, 2]));
        assert_eq!(parsed.extra_params["future_ocr_option"], true);
        assert_eq!(
            parsed.extra_params["extra_body"],
            json!({"provider_option": "value"})
        );
        assert_eq!(parsed.extra_params.len(), 2);
    }
}
