use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use super::OcrClient;
use super::error::{OcrError, OcrRequestError};
use super::hooks::OcrDuringCallRequest;
use super::types::{LiteLLMOcrRequest, OcrDocument};

#[derive(Debug, Deserialize)]
pub(crate) struct ParsedProviderParams<T> {
    #[serde(flatten)]
    pub known: T,
    #[serde(default, flatten)]
    pub extra_params: Map<String, Value>,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) fn _prepare_ocr_request<T: DeserializeOwned>(
    request: &LiteLLMOcrRequest,
) -> Result<ParsedProviderParams<T>, OcrRequestError> {
    super::wire::decode_request_value(
        Value::Object(request.optional_params.clone()),
        "optional_params",
    )
}

pub(crate) fn merge_extra_params<B: Serialize>(
    body: &B,
    extra_params: Map<String, Value>,
) -> Result<Value, OcrRequestError> {
    let Value::Object(fields) =
        serde_json::to_value(body).map_err(|_| OcrRequestError::RequestField {
            path: "body".into(),
        })?
    else {
        return Err(OcrRequestError::RequestField {
            path: "body".into(),
        });
    };
    let extra_body = extra_params
        .get("extra_body")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect::<Map<String, Value>>();
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                extra_params
                    .into_iter()
                    .filter(|(name, _)| name != "extra_body"),
            )
            .chain(extra_body)
            .collect(),
    ))
}

pub(crate) async fn transform_request_body<B>(
    client: &OcrClient,
    request: &LiteLLMOcrRequest,
    url: &str,
    headers: &[(String, String)],
    body: B,
    validate: impl FnOnce(&B) -> Result<(), OcrRequestError>,
) -> Result<reqwest::Request, OcrError>
where
    B: Serialize + DeserializeOwned,
{
    let body = if request.hooks.has_guardrails() {
        let changed = request
            .hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: request.adapter.provider().as_str().into(),
                url: url.into(),
                body: serde_json::to_value(body).map_err(|_| OcrRequestError::RequestField {
                    path: "body".into(),
                })?,
            })
            .await?;
        let body = OcrWireBody::<B>::decode(changed.body)?;
        validate(&body.body)?;
        body
    } else {
        OcrWireBody {
            body,
            extra: Map::new(),
        }
    };
    build_http_request(client, request, url, headers, &body)
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
) -> Result<OcrDocument, OcrError> {
    if !request.hooks.has_guardrails() {
        return Ok(request.document.clone());
    }
    let changed = request
        .hooks
        .during_call(OcrDuringCallRequest {
            model: request.model.clone(),
            custom_llm_provider: request.adapter.provider().as_str().into(),
            url: url.into(),
            body: serde_json::to_value(&request.document).map_err(|_| {
                OcrRequestError::RequestField {
                    path: "document".into(),
                }
            })?,
        })
        .await?;
    super::wire::decode_request_value(changed.body, "guardrail.document").map_err(OcrError::from)
}

#[derive(Serialize)]
struct OcrWireBody<B> {
    #[serde(flatten)]
    body: B,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

impl<B: Serialize + DeserializeOwned> OcrWireBody<B> {
    fn decode(value: Value) -> Result<Self, OcrRequestError> {
        let body: B = super::wire::decode_request_value(value.clone(), "guardrail.body")?;
        let Value::Object(fields) = value else {
            return Err(OcrRequestError::RequestField {
                path: "guardrail.body".into(),
            });
        };
        let known = serde_json::to_value(&body).map_err(|_| OcrRequestError::RequestField {
            path: "guardrail.body".into(),
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
