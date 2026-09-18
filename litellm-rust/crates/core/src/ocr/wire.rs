use std::{collections::BTreeMap, time::Duration};

use litellm_auth::InputSource;
use litellm_llms::base_llm::ocr::{
    error::Error,
    transformation::{OcrDocument, decode_request_value},
};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::ocr::types::{LiteLLMOcrRequest, OcrConnectionInputs, OcrDocumentInput};

pub fn consumed_optional_params(
    model: &str,
    provider: Option<&str>,
) -> Result<Vec<litellm_core_utils::call_arguments::ArgumentSpec>, Error> {
    let specs = crate::ocr::arguments::consumed_optional_params(model, provider)?;
    Ok(consumed_optional_param_names(model, provider)?
        .into_iter()
        .map(|name| litellm_core_utils::call_arguments::ArgumentSpec {
            name,
            secret: specs.iter().any(|spec| spec.name == name && spec.secret),
        })
        .collect())
}

pub fn consumed_optional_param_names(
    model: &str,
    provider: Option<&str>,
) -> Result<Vec<&'static str>, Error> {
    let names = crate::ocr::arguments::consumed_optional_param_names(model, provider)?;
    let (_, config) = super::provider_config::resolve_provider_config(model, provider)?;
    if config == super::provider_config::OcrConfigKind::VertexDeepSeek {
        return Ok(names
            .into_iter()
            .chain(["stream", "temperature", "max_tokens", "top_p", "n", "stop"])
            .collect());
    }
    Ok(names)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrWireRequest<D = Value> {
    pub model: String,
    pub document: D,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    #[serde(default)]
    pub optional_params: Map<String, Value>,
    #[serde(default)]
    pub input_sources: BTreeMap<String, InputSource>,
    pub timeout_seconds: Option<f64>,
}

pub fn decode_request(wire: OcrWireRequest) -> Result<LiteLLMOcrRequest, Error> {
    decode_request_input(OcrWireRequest {
        model: wire.model,
        document: decode_document(wire.document)?,
        api_key: wire.api_key,
        api_base: wire.api_base,
        custom_llm_provider: wire.custom_llm_provider,
        extra_headers: wire.extra_headers,
        optional_params: wire.optional_params,
        input_sources: wire.input_sources,
        timeout_seconds: wire.timeout_seconds,
    })
}

pub fn decode_request_input<D: Into<OcrDocumentInput>>(
    wire: OcrWireRequest<D>,
) -> Result<LiteLLMOcrRequest, Error> {
    let timeout = wire
        .timeout_seconds
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds).map_err(|_| Error::RequestField {
                path: "timeout_seconds".into(),
            })
        })
        .transpose()?;
    LiteLLMOcrRequest::from_inputs(
        wire.model,
        wire.document,
        wire.custom_llm_provider.as_deref(),
        wire.optional_params.into(),
        OcrConnectionInputs {
            api_key: wire.api_key,
            api_base: wire.api_base,
            extra_headers: wire.extra_headers.unwrap_or_default(),
            timeout,
            input_sources: wire.input_sources,
        },
    )
}

pub fn decode_document(value: Value) -> Result<OcrDocument, Error> {
    let kind = value.get("type").and_then(Value::as_str);
    if matches!(kind, Some("document_url")) && value.get("document_url").is_none()
        || matches!(kind, Some("image_url")) && value.get("image_url").is_none()
    {
        return Err(Error::MissingDocumentUrl);
    }
    decode_request_value(value, "document")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::ocr::arguments::is_supported_request;

    #[rstest]
    #[case::omitted(json!({"type":"document_url", "document_url":"https://example.com/a.pdf"}))]
    #[case::null(json!({"type":"document_url", "document_url":"https://example.com/a.pdf", "document_name":null}))]
    fn ocr_contract_optional_document_name(#[case] document: Value) {
        let decoded = decode_document(document).unwrap();
        assert_eq!(decoded.source(), "https://example.com/a.pdf");
    }

    #[rstest]
    #[case::non_object(json!([]), "document")]
    #[case::missing_type(json!({"document_url":"https://example.com/a.pdf"}), "document")]
    #[case::unsupported_type(json!({"type":"text"}), "type")]
    #[case::missing_document_url(json!({"type":"document_url"}), "Document URL")]
    #[case::missing_image_url(json!({"type":"image_url"}), "Document URL")]
    fn ocr_contract_malformed_document_is_bad_request(
        #[case] document: Value,
        #[case] field: &str,
    ) {
        let error = decode_document(document).unwrap_err();
        assert!(matches!(
            error,
            Error::RequestField { .. } | Error::MissingDocumentUrl
        ));
        assert_eq!(error.http_status_code(), Some(400));
        assert!(error.to_string().contains(field), "{error}");
    }

    #[test]
    fn option_projection_is_provider_specific_and_excludes_opaque_fields() {
        let mistral = consumed_optional_param_names("mistral/model", None).unwrap();
        assert!(mistral.contains(&"pages"));
        assert!(mistral.contains(&"req_format"));
        assert!(!mistral.contains(&"vertex_project"));
        assert!(!mistral.contains(&"opaque_extension"));
        let vertex = consumed_optional_param_names("vertex_ai/deepseek-ocr", None).unwrap();
        assert!(vertex.contains(&"temperature"));
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
            assert!(matches!(
                decode_document(document),
                Err(Error::MissingDocumentUrl)
            ));
        }
    }
}
