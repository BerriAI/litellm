use std::collections::BTreeMap;
use std::time::Duration;

use litellm_auth::InputSource;
use litellm_core::call_arguments::CallArguments;
use litellm_core::ocr::{LiteLLMOcrRequest, OcrCredentialInputs, OcrDocument};
use serde_json::{Map, Value};

pub(super) struct BridgeOcrRequest {
    pub model: String,
    pub document: Value,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: CallArguments,
    pub input_sources: BTreeMap<String, InputSource>,
    pub timeout_seconds: Option<f64>,
}

impl TryFrom<BridgeOcrRequest> for LiteLLMOcrRequest {
    type Error = litellm_core::ocr::Error;

    fn try_from(request: BridgeOcrRequest) -> Result<Self, Self::Error> {
        let api_key_source = source_for(&request.input_sources, "api_key");
        let api_base_source = source_for(&request.input_sources, "api_base");
        let extra_headers_source = source_for(&request.input_sources, "extra_headers");
        let timeout = request
            .timeout_seconds
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds).map_err(|_| Self::Error::RequestField {
                    path: "timeout_seconds".into(),
                })
            })
            .transpose()?;
        let headers = request
            .extra_headers
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| {
                value
                    .as_str()
                    .map(|value| (name.clone(), value.to_string()))
                    .ok_or_else(|| Self::Error::RequestField {
                        path: format!("extra_headers.{name}"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let core_request = LiteLLMOcrRequest::new(
            request.model,
            OcrDocument::try_from(request.document)?,
            request.custom_llm_provider.as_deref(),
            request.optional_params,
        )?;
        let transport =
            core_request
                .transport
                .clone()
                .with_overrides(headers, extra_headers_source, timeout);
        Ok(core_request.with_connection_inputs(
            OcrCredentialInputs::new(
                request.api_key,
                api_key_source,
                request.api_base,
                api_base_source,
            ),
            transport,
            request.input_sources,
        ))
    }
}

fn source_for(sources: &BTreeMap<String, InputSource>, name: &str) -> InputSource {
    sources.get(name).copied().unwrap_or_default()
}
