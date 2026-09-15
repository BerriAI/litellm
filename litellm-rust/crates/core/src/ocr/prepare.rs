use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use super::Error;
use super::OcrClient;
use super::hooks::OcrDuringCallRequest;
use super::types::{LiteLLMOcrRequest, OcrDocument};

pub(crate) use crate::params::{ParsedProviderParams, merge_extra_params};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn _prepare_ocr_request<T: DeserializeOwned>(
    request: &LiteLLMOcrRequest,
) -> Result<ParsedProviderParams<T>, Error> {
    super::wire::decode_request_value(
        Value::Object(request.optional_params.provider_params().into()),
        "optional_params",
    )
}

pub(crate) async fn transform_request_body<B>(
    client: &OcrClient,
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
    retains_document: bool,
    body: B,
    validate: impl Fn(&B) -> Result<(), Error>,
) -> Result<reqwest::Request, Error>
where
    B: Serialize + DeserializeOwned,
{
    let extras = request
        .optional_params
        .without(request.config.get_supported_ocr_params(&request.model));
    let composed = merge_extra_params(&body, extras)?;
    let composed = OcrWireBody::<B>::decode(composed, "body")?;
    validate(&composed.body)?;
    let (body, headers) = if request.hooks.intercepts_requests() {
        let body = serde_json::to_value(composed).map_err(|_| Error::RequestField {
            path: "body".into(),
        })?;
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
                custom_llm_provider: request.config.provider().as_str().into(),
                url: url.into(),
                headers: headers.to_vec(),
                body,
                retained_fields,
            })
            .await?;
        let body = OcrWireBody::<B>::decode(changed.body, "guardrail.body")?;
        validate(&body.body)?;
        (body, changed.headers)
    } else {
        (composed, headers.to_vec())
    };
    build_http_request(client, request, url, &headers, &body)
}

pub(crate) fn build_http_request<B: Serialize>(
    client: &OcrClient,
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: &B,
) -> Result<reqwest::Request, Error> {
    let builder = client
        .provider_http()
        .post(url)
        .json(body)
        .timeout(request.connection.timeout);
    crate::http_utils::with_headers(builder, headers, crate::http_utils::HeaderPolicy::All)
        .build()
        .map_err(crate::transport::Error::from)
        .map_err(Error::from)
}

pub(crate) async fn guardrail_document(
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
) -> Result<(OcrDocument, Vec<(String, String)>), Error> {
    if !request.hooks.intercepts_requests() {
        return Ok((request.document.clone(), headers.to_vec()));
    }
    let changed = request
        .hooks
        .during_call(OcrDuringCallRequest {
            model: request.model.clone(),
            custom_llm_provider: request.config.provider().as_str().into(),
            url: url.into(),
            headers: headers.to_vec(),
            body: serde_json::to_value(&request.document).map_err(|_| Error::RequestField {
                path: "document".into(),
            })?,
            retained_fields: Vec::new(),
        })
        .await?;
    let document = super::wire::decode_request_value(changed.body, "guardrail.document")?;
    Ok((document, changed.headers))
}

#[derive(Serialize)]
struct OcrWireBody<B> {
    #[serde(flatten)]
    body: B,
    #[serde(flatten)]
    extra: crate::params::OpaqueParams,
}

impl<B: Serialize + DeserializeOwned> OcrWireBody<B> {
    fn decode(value: Value, prefix: &str) -> Result<Self, Error> {
        let body: B = super::wire::decode_request_value(value.clone(), prefix)?;
        let Value::Object(fields) = value else {
            return Err(Error::RequestField {
                path: prefix.into(),
            });
        };
        let known = serde_json::to_value(&body).map_err(|_| Error::RequestField {
            path: prefix.into(),
        })?;
        let extra = fields
            .into_iter()
            .filter(|(key, _)| known.get(key).is_none())
            .collect();
        Ok(Self { body, extra })
    }
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
