use std::sync::Arc;
use std::time::Duration;

use litellm_auth::InputSource;
use litellm_core::ocr::{LiteLLMOcrRequest, OcrCredentialInputs, OcrDocument};
use litellm_core::params::OpaqueParams;
use serde_json::{Map, Value};

use crate::integrations::custom_guardrail::CustomGuardrail;
use crate::integrations::custom_logger::CustomLogger;
use crate::integrations::types::RequestMetadata;

pub struct OcrRequest<'a> {
    pub model: &'a str,
    pub document: Value,
    pub api_key: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub custom_llm_provider: Option<&'a str>,
    pub extra_headers: Option<Map<String, Value>>,
    pub optional_params: OpaqueParams,
    pub timeout: Option<Duration>,
    pub callbacks: Vec<Arc<dyn CustomLogger>>,
    pub guardrails: Vec<Arc<dyn CustomGuardrail>>,
    pub request_metadata: RequestMetadata,
    pub litellm_call_id: Option<&'a str>,
}

impl TryFrom<OcrRequest<'_>> for LiteLLMOcrRequest {
    type Error = litellm_core::ocr::Error;

    fn try_from(request: OcrRequest<'_>) -> Result<Self, Self::Error> {
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
            request.model.to_string(),
            OcrDocument::try_from(request.document)?,
            request.custom_llm_provider,
            request.optional_params.into_inner().into(),
        )?;
        let transport = core_request.transport.clone().with_overrides(
            headers,
            InputSource::Deployment,
            request.timeout,
        );
        Ok(core_request.with_connection_inputs(
            OcrCredentialInputs::new(
                request.api_key.map(str::to_string),
                InputSource::Deployment,
                request.api_base.map(str::to_string),
                InputSource::Deployment,
            ),
            transport,
            Default::default(),
        ))
    }
}
