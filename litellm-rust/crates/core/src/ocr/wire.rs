use std::collections::BTreeMap;
use std::time::Duration;

use super::types::{LiteLLMOcrRequest, OcrDocument, OcrTransportConfig};
use crate::call_arguments::{ArgumentSpec, CallArguments};
use litellm_auth::InputSource;
use serde::{
    Deserialize,
    de::{DeserializeOwned, IntoDeserializer},
};
use serde_json::{Map, Value};

const COMMON_OPTION_FIELDS: &[&str] = &["req_format", "extra_body", "max_response_bytes"];
pub const BOUND_FIELDS: &[&str] = &["model", "document", "timeout", "input_sources"];
const AZURE_AUTH_OPTION_FIELDS: &[&str] = &[
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
    "enable_azure_ad_token_refresh",
];
const VERTEX_AUTH_OPTION_FIELDS: &[&str] = &[
    "vertex_credentials",
    "vertex_ai_credentials",
    "vertex_project",
    "vertex_ai_project",
    "vertex_location",
    "vertex_ai_location",
];

#[derive(Debug)]
pub struct DecodedOcrResponse<T> {
    pub data: T,
    pub native: Option<Map<String, Value>>,
    pub text: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrWireRequest {
    pub model: String,
    pub document: Value,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    #[serde(default)]
    pub optional_params: CallArguments,
    #[serde(default)]
    pub input_sources: BTreeMap<String, InputSource>,
    pub timeout_seconds: Option<f64>,
}

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    super::provider_config::resolve_provider_config(model, custom_llm_provider).is_ok()
}

pub fn consumed_optional_param_names(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<&'static str>, crate::ocr::Error> {
    use super::provider_config::OcrConfigKind;

    let (model, config) =
        super::provider_config::resolve_provider_config(model, custom_llm_provider)?;
    let provider_fields = config.get_supported_ocr_params(&model);
    let auth_fields: &[&str] = match config {
        OcrConfigKind::AzureAi
        | OcrConfigKind::AzureDocumentIntelligence
        | OcrConfigKind::AzureCohere => AZURE_AUTH_OPTION_FIELDS,
        OcrConfigKind::VertexAi | OcrConfigKind::VertexDeepSeek => VERTEX_AUTH_OPTION_FIELDS,
        _ => &[],
    };
    Ok(COMMON_OPTION_FIELDS
        .iter()
        .chain(provider_fields)
        .chain(auth_fields)
        .copied()
        .collect())
}

pub fn consumed_optional_params(
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<Vec<ArgumentSpec>, crate::ocr::Error> {
    consumed_optional_param_names(model, custom_llm_provider).map(|names| {
        names
            .into_iter()
            .map(|name| ArgumentSpec {
                name,
                secret: matches!(
                    name,
                    "azure_ad_token"
                        | "client_secret"
                        | "azure_federated_token_file"
                        | "vertex_credentials"
                        | "vertex_ai_credentials"
                ),
            })
            .collect()
    })
}

pub fn decode_request(wire: OcrWireRequest) -> Result<LiteLLMOcrRequest, crate::ocr::Error> {
    let api_key_source = source_for(&wire.input_sources, "api_key");
    let api_base_source = source_for(&wire.input_sources, "api_base");
    let extra_headers_source = source_for(&wire.input_sources, "extra_headers");
    let document = decode_document(wire.document)?;
    let headers = wire
        .extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            let value = value
                .as_str()
                .ok_or_else(|| crate::ocr::Error::RequestField {
                    path: format!("extra_headers.{name}"),
                })?;
            Ok((name, value.to_string()))
        })
        .collect::<Result<Vec<_>, crate::ocr::Error>>()?;
    let timeout = wire
        .timeout_seconds
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds).map_err(|_| crate::ocr::Error::RequestField {
                path: "timeout_seconds".into(),
            })
        })
        .transpose()?;
    let defaults = OcrTransportConfig::default();
    let max_response_bytes = wire
        .optional_params
        .get("max_response_bytes")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0 && *value <= defaults.max_response_bytes)
                .ok_or_else(|| crate::ocr::Error::RequestField {
                    path: "max_response_bytes".into(),
                })
        })
        .transpose()?
        .unwrap_or(defaults.max_response_bytes);
    let request = LiteLLMOcrRequest::new(
        wire.model,
        document,
        wire.custom_llm_provider.as_deref(),
        wire.optional_params
            .into_iter()
            .filter(|(name, _)| name != "max_response_bytes")
            .collect(),
    )?;
    let credentials = super::types::OcrCredentialInputs {
        api_key: nonblank(wire.api_key)
            .map(|value| litellm_auth::Sourced::new(value, api_key_source)),
        dynamic_api_key: None,
        api_base: nonblank(wire.api_base)
            .map(|value| litellm_auth::Sourced::new(value, api_base_source)),
        dynamic_api_base: None,
    };
    let transport = OcrTransportConfig {
        extra_headers: headers,
        extra_headers_source,
        timeout: timeout.unwrap_or(defaults.timeout),
        max_download_bytes: defaults.max_download_bytes,
        max_response_bytes,
        poll_timeout: defaults.poll_timeout,
    };
    Ok(LiteLLMOcrRequest {
        credentials,
        transport,
        input_sources: wire.input_sources,
        ..request
    })
}

fn decode_document(value: Value) -> Result<OcrDocument, crate::ocr::Error> {
    let kind = value.get("type").and_then(Value::as_str);
    let missing_url = matches!(kind, Some("document_url")) && value.get("document_url").is_none()
        || matches!(kind, Some("image_url")) && value.get("image_url").is_none();
    if missing_url {
        return Err(crate::ocr::Error::MissingDocumentUrl);
    }
    decode_request_value(value, "document")
}

fn source_for(sources: &BTreeMap<String, InputSource>, name: &str) -> InputSource {
    sources.get(name).copied().unwrap_or_default()
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
pub fn decode_request_value<T: DeserializeOwned>(
    value: Value,
    prefix: &str,
) -> Result<T, crate::ocr::Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        crate::ocr::Error::RequestField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub(crate) fn decode_response_value<T: DeserializeOwned>(
    value: Value,
    prefix: &str,
) -> Result<T, crate::ocr::Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        crate::ocr::Error::ResponseField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub fn decode_response<T: DeserializeOwned>(
    bytes: &[u8],
    native: bool,
) -> Result<DecodedOcrResponse<T>, crate::ocr::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let data = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        crate::ocr::Error::ResponseField {
            path: error.path().to_string(),
        }
    })?;
    deserializer
        .end()
        .map_err(|_| crate::ocr::Error::ResponseField {
            path: "response".into(),
        })?;
    let native = if native {
        Some(
            serde_json::from_slice(bytes).map_err(|_| crate::ocr::Error::ResponseField {
                path: "response".into(),
            })?,
        )
    } else {
        None
    };
    Ok(DecodedOcrResponse {
        data,
        native,
        text: String::from_utf8_lossy(bytes).into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_selects_consumed_values_without_serializing_host_objects() {
        let fields = consumed_optional_params("mistral/model", None).unwrap();
        use crate::call_arguments::should_project;
        assert!(should_project("future_option", &fields, BOUND_FIELDS));
        assert!(should_project("extra_body", &fields, BOUND_FIELDS));
        assert!(should_project("id", &fields, BOUND_FIELDS));
        assert!(!should_project("metadata", &fields, BOUND_FIELDS));
        assert!(!should_project("callbacks", &fields, BOUND_FIELDS));
        assert!(!should_project("api_key", &fields, BOUND_FIELDS));
        assert!(!should_project("document", &fields, BOUND_FIELDS));
    }

    #[test]
    fn option_projection_is_provider_specific_and_excludes_opaque_fields() {
        let mistral = consumed_optional_param_names("mistral/model", None).unwrap();
        assert!(mistral.contains(&"pages"));
        assert!(mistral.contains(&"req_format"));
        assert!(!mistral.contains(&"vertex_project"));
        assert!(!mistral.contains(&"opaque_extension"));

        let vertex = consumed_optional_param_names("vertex_ai/deepseek-ocr", None).unwrap();
        assert!(!vertex.contains(&"temperature"));
        assert!(vertex.contains(&"vertex_credentials"));
        assert!(!vertex.contains(&"pages"));
    }

    #[test]
    fn optional_param_metadata_marks_only_credentials_as_secret() {
        let azure = consumed_optional_params("model", Some("azure_ai")).unwrap();
        assert!(
            azure
                .iter()
                .any(|spec| spec.name == "client_secret" && spec.secret)
        );
        assert!(
            azure
                .iter()
                .any(|spec| spec.name == "tenant_id" && !spec.secret)
        );
        let vertex = consumed_optional_params("deepseek-ocr", Some("vertex_ai")).unwrap();
        assert!(
            vertex
                .iter()
                .any(|spec| spec.name == "vertex_credentials" && spec.secret)
        );
        assert!(
            vertex
                .iter()
                .any(|spec| spec.name == "vertex_project" && !spec.secret)
        );
    }

    #[test]
    fn activation_includes_migrated_providers() {
        assert!(is_supported_request("model", Some("mistral")));
        assert!(is_supported_request("pixtral-12b", Some("azure_ai")));
        assert!(is_supported_request(
            "documentintelligence/prebuilt-read",
            Some("azure_ai")
        ));
        assert!(is_supported_request("parse-v3", Some("reducto")));
        assert!(is_supported_request("parse-legacy", Some("reducto")));
        assert!(is_supported_request("mistral-ocr", Some("vertex_ai")));
        assert!(is_supported_request("deepseek-ocr", Some("vertex_ai")));
    }

    #[test]
    fn missing_document_source_has_a_typed_public_error() {
        for document in [
            serde_json::json!({"type": "document_url"}),
            serde_json::json!({"type": "image_url"}),
        ] {
            let wire = serde_json::from_value(serde_json::json!({
                "model": "mistral/model",
                "document": document,
            }))
            .unwrap();
            let error = decode_request(wire).err().expect("missing document URL");
            assert_eq!(error, crate::ocr::Error::MissingDocumentUrl);
            let error = crate::Error::from(error);
            assert!(matches!(
                error,
                crate::Error::Ocr(crate::ocr::Error::MissingDocumentUrl)
            ));
        }
    }
}
