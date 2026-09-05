use crate::Error;
use serde_json::{Map, Value};

use super::types::{OcrAuthentication, OcrRequestData, OcrResponseData};
use crate::auth::{AuthPreflight, CredentialSpec, SecretString};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrAuthStrategy {
    Bearer,
    Header(&'static str),
}

impl OcrAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrResponseHandling {
    Json,
    AzureDocumentIntelligencePoll,
}

pub trait OcrProviderConfig: Sync {
    fn supported_ocr_params(&self) -> &'static [&'static str];

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        let mut mapped_params = Map::new();
        for (param, value) in non_default_params {
            if self.supported_ocr_params().contains(&param.as_str()) {
                mapped_params.insert(param.clone(), value.clone());
            }
        }
        mapped_params
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: Value,
        optional_params: Map<String, Value>,
    ) -> Result<OcrRequestData, Error>;

    fn transform_ocr_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> Result<OcrResponseData, Error>;

    fn transform_ocr_response_with_params(
        &self,
        model: &str,
        response_json: Value,
        _optional_params: &Map<String, Value>,
    ) -> Result<OcrResponseData, Error> {
        self.transform_ocr_response(model, response_json)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Vec<(String, String)>, Error> {
        let strategy = self.auth_strategy();
        if crate::http_utils::has_header(&headers, strategy.header_name()) {
            return Ok(headers);
        }
        let api_key = self.resolve_api_key(api_key, env_lookup)?;
        let auth_header = match strategy {
            OcrAuthStrategy::Bearer => ("Authorization".to_string(), format!("Bearer {api_key}")),
            OcrAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        Ok(std::iter::once(auth_header).chain(headers).collect())
    }

    fn select_auth(
        &self,
        api_key: Option<&str>,
        headers: Vec<(String, String)>,
        connection: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<AuthPreflight<OcrAuthentication>, Error> {
        if connection.values().any(|value| !value.is_null()) {
            return Ok(AuthPreflight::Declined(
                "credential configuration requires an unimplemented adapter",
            ));
        }
        let resolved_key = self.resolve_api_key(api_key, env_lookup).ok();
        let headers = self.validate_environment(headers, api_key, env_lookup)?;
        let headers = headers
            .into_iter()
            .map(|(name, value)| {
                let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::Auth("invalid provider header name".into()))?;
                let value = reqwest::header::HeaderValue::from_str(&value)
                    .map_err(|_| Error::Auth("invalid provider header value".into()))?;
                Ok((name, value))
            })
            .collect::<Result<reqwest::header::HeaderMap, Error>>()?;
        let name = if headers.contains_key(reqwest::header::AUTHORIZATION) {
            reqwest::header::AUTHORIZATION
        } else {
            reqwest::header::HeaderName::from_bytes(self.auth_strategy().header_name().as_bytes())
                .map_err(|_| Error::Auth("invalid auth header name".into()))?
        };
        let value = headers
            .get(&name)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| Error::Auth("provider authentication header is missing".into()))?;
        Ok(AuthPreflight::Ready(OcrAuthentication {
            credential: CredentialSpec::Header {
                name,
                value: SecretString::new(value),
            },
            headers,
            api_key: resolved_key,
        }))
    }

    fn auth_strategy(&self) -> OcrAuthStrategy {
        OcrAuthStrategy::Bearer
    }

    fn requires_data_uri_document(&self) -> bool {
        false
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::Json
    }
}
