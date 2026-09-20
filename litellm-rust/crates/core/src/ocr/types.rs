use std::{collections::BTreeMap, path::PathBuf, time::Duration};

use bytes::Bytes;
use litellm_auth::{InputSource, SecretValue, TokenProviderHandle};
use litellm_core_utils::call_arguments::CallArguments;
use litellm_llms::base_llm::ocr::{
    error::Error,
    transformation::{
        OcrCredentialInputs, OcrDocument, OcrResponseFormat, OcrTransportConfig, response_format,
    },
};
use serde_json::{Map, Value};

use super::provider_config::{OcrConfigKind, resolve_provider_config};

#[derive(Clone, Debug, PartialEq)]
pub enum OcrDocumentInput {
    Document(OcrDocument),
    Path {
        path: PathBuf,
        mime_type: Option<String>,
    },
    Bytes {
        bytes: Bytes,
        file_name: Option<String>,
        mime_type: Option<String>,
    },
    HostReader {
        mime_type: Option<String>,
    },
}

impl From<OcrDocument> for OcrDocumentInput {
    fn from(document: OcrDocument) -> Self {
        Self::Document(document)
    }
}

impl From<PathBuf> for OcrDocumentInput {
    fn from(path: PathBuf) -> Self {
        Self::Path {
            path,
            mime_type: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OcrFileContent {
    pub bytes: Bytes,
    pub file_name: Option<String>,
}

/// Caller-supplied connection overrides for a [`LiteLLMOcrRequest`], in the
/// shape hosts receive them: JSON-ish headers, optional timeout, optional
/// credentials, and per-field provenance in `input_sources`.
#[derive(Clone, Debug, Default)]
pub struct OcrConnectionInputs {
    pub api_key: Option<SecretValue>,
    pub api_base: Option<String>,
    pub extra_headers: Map<String, Value>,
    pub timeout: Option<Duration>,
    pub input_sources: BTreeMap<String, InputSource>,
}

impl OcrConnectionInputs {
    fn source(&self, name: &str) -> InputSource {
        self.input_sources.get(name).copied().unwrap_or_default()
    }

    fn header_pairs(&self) -> Result<Vec<(String, String)>, Error> {
        self.extra_headers
            .iter()
            .map(|(name, value)| {
                value
                    .as_str()
                    .map(|value| (name.clone(), value.to_string()))
                    .ok_or_else(|| Error::RequestField {
                        path: format!("extra_headers.{name}"),
                    })
            })
            .collect()
    }
}

pub struct LiteLLMOcrRequest<D = OcrDocumentInput> {
    pub model: String,
    pub document: D,
    pub credentials: OcrCredentialInputs,
    pub transport: OcrTransportConfig,
    pub optional_params: CallArguments,
    pub input_sources: BTreeMap<String, InputSource>,
    pub azure_ad_token_provider: Option<TokenProviderHandle>,
    pub(crate) config: OcrConfigKind,
}

impl LiteLLMOcrRequest {
    pub fn new(
        model: String,
        document: impl Into<OcrDocumentInput>,
        custom_llm_provider: Option<&str>,
        optional_params: CallArguments,
    ) -> Result<Self, Error> {
        let (model, config) = resolve_provider_config(&model, custom_llm_provider)?;
        let default_transport = OcrTransportConfig::default();
        let max_response_bytes = optional_params
            .get("max_response_bytes")
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .filter(|value| *value > 0 && *value <= default_transport.max_response_bytes)
                    .ok_or_else(|| Error::RequestField {
                        path: "max_response_bytes".into(),
                    })
            })
            .transpose()?
            .unwrap_or(default_transport.max_response_bytes);
        let transport = OcrTransportConfig {
            max_response_bytes,
            ..default_transport
        };
        let optional_params = optional_params
            .into_iter()
            .filter(|(name, _)| name != "max_response_bytes")
            .collect();

        Ok(Self {
            model,
            document: document.into(),
            credentials: OcrCredentialInputs::default(),
            transport,
            optional_params,
            input_sources: BTreeMap::new(),
            azure_ad_token_provider: None,
            config,
        })
    }
}

impl<D> LiteLLMOcrRequest<D> {
    pub fn map_document<T, E>(
        self,
        map: impl FnOnce(D) -> Result<T, E>,
    ) -> Result<LiteLLMOcrRequest<T>, E> {
        Ok(LiteLLMOcrRequest {
            model: self.model,
            document: map(self.document)?,
            credentials: self.credentials,
            transport: self.transport,
            optional_params: self.optional_params,
            input_sources: self.input_sources,
            azure_ad_token_provider: self.azure_ad_token_provider,
            config: self.config,
        })
    }

    pub fn with_document<T>(self, document: T) -> LiteLLMOcrRequest<T> {
        LiteLLMOcrRequest {
            model: self.model,
            document,
            credentials: self.credentials,
            transport: self.transport,
            optional_params: self.optional_params,
            input_sources: self.input_sources,
            azure_ad_token_provider: self.azure_ad_token_provider,
            config: self.config,
        }
    }

    pub(crate) fn response_format(&self) -> Result<OcrResponseFormat, Error> {
        response_format(&self.optional_params)
    }

    pub fn provider_name(&self) -> &'static str {
        self.config.provider().into()
    }

    pub fn with_connection_inputs(
        self,
        credentials: OcrCredentialInputs,
        transport: OcrTransportConfig,
        input_sources: BTreeMap<String, InputSource>,
    ) -> Self {
        Self {
            credentials,
            transport,
            input_sources,
            ..self
        }
    }
}

impl LiteLLMOcrRequest {
    /// Builds a request from host-shaped inputs in one step: provider
    /// resolution, optional-param validation, header/timeout overrides and
    /// sourced credentials. Hosts should prefer this over sequencing
    /// [`Self::new`], [`OcrTransportConfig::with_overrides`] and
    /// [`Self::with_connection_inputs`] by hand.
    pub fn from_inputs(
        model: String,
        document: impl Into<OcrDocumentInput>,
        custom_llm_provider: Option<&str>,
        optional_params: CallArguments,
        connection: OcrConnectionInputs,
    ) -> Result<Self, Error> {
        let request = Self::new(model, document, custom_llm_provider, optional_params)?;
        let transport = request.transport.clone().with_overrides(
            connection.header_pairs()?,
            connection.source("extra_headers"),
            connection.timeout,
        );
        let (api_key_source, api_base_source) =
            (connection.source("api_key"), connection.source("api_base"));
        let credentials = OcrCredentialInputs::new(
            connection.api_key,
            api_key_source,
            connection.api_base,
            api_base_source,
        );
        Ok(request.with_connection_inputs(credentials, transport, connection.input_sources))
    }
}

pub(crate) type ResolvedOcrRequest = LiteLLMOcrRequest<OcrDocument>;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn document() -> OcrDocument {
        OcrDocument::try_from(
            json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        )
        .unwrap()
    }

    #[test]
    fn connection_inputs_debug_hides_the_api_key() {
        let inputs = OcrConnectionInputs {
            api_key: Some(SecretValue::new("caller-api-key")),
            ..OcrConnectionInputs::default()
        };

        assert!(!format!("{inputs:?}").contains("caller-api-key"));
    }

    #[test]
    fn from_inputs_applies_connection_overrides_with_field_sources() {
        let request = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs {
                api_key: Some(SecretValue::new(" key ")),
                api_base: Some("".into()),
                extra_headers: json!({"x-a": "1"}).as_object().unwrap().clone(),
                timeout: Some(Duration::from_secs(7)),
                input_sources: [
                    ("api_key".to_string(), InputSource::Request),
                    ("extra_headers".to_string(), InputSource::Request),
                ]
                .into(),
            },
        )
        .unwrap();

        let api_key = request.credentials.api_key.as_ref().unwrap();
        assert_eq!(api_key.value().expose(), "key");
        assert_eq!(api_key.source(), InputSource::Request);
        assert!(request.credentials.api_base.is_none());
        assert_eq!(
            request.transport.extra_headers,
            vec![("x-a".to_string(), "1".to_string())]
        );
        assert_eq!(request.transport.extra_headers_source, InputSource::Request);
        assert_eq!(request.transport.timeout, Some(Duration::from_secs(7)));
        assert_eq!(request.input_sources.len(), 2);

        let defaulted = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs::default(),
        )
        .unwrap();
        assert_eq!(
            defaulted.transport.timeout,
            OcrTransportConfig::default().timeout
        );
        assert_eq!(
            defaulted.transport.extra_headers_source,
            InputSource::Deployment
        );
    }

    #[test]
    fn from_inputs_rejects_non_string_header_values_by_path() {
        let Err(error) = LiteLLMOcrRequest::from_inputs(
            "mistral/model".into(),
            document(),
            None,
            Default::default(),
            OcrConnectionInputs {
                extra_headers: json!({"x-a": 1}).as_object().unwrap().clone(),
                ..Default::default()
            },
        ) else {
            panic!("non-string header value accepted");
        };
        assert!(matches!(
            error,
            Error::RequestField { ref path } if path == "extra_headers.x-a"
        ));
    }
}
