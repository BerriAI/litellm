use async_trait::async_trait;

use crate::Error;
use crate::auth::{AuthHeaderKind, Environment, ResolvedAuth};
use serde_json::{Map, Value};

use super::types::{OcrRequestData, OcrResponseData};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrResponseHandling {
    Json,
    AzureDocumentIntelligencePoll,
}

#[async_trait]
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
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<String, Error>;

    fn forwarded_auth(&self, headers: &[(String, String)]) -> Option<ResolvedAuth> {
        let kind = self.auth_header_kind();
        headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(kind.header_name()))
            .and_then(|(_, value)| ResolvedAuth::from_header(kind, value))
    }

    async fn resolve_auth(
        &self,
        api_key: Option<&str>,
        _optional_params: &Map<String, Value>,
        environment: &dyn Environment,
    ) -> Result<ResolvedAuth, Error> {
        let env_lookup = |name: &str| environment.get(name);
        Ok(ResolvedAuth::from_credential(
            self.auth_header_kind(),
            self.resolve_api_key(api_key, &env_lookup)?,
        ))
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn validate_environment(
        &self,
        headers: Vec<(String, String)>,
        api_key: Option<&str>,
        optional_params: &Map<String, Value>,
        environment: &dyn Environment,
    ) -> Result<(Vec<(String, String)>, ResolvedAuth), Error> {
        let auth = match self.forwarded_auth(&headers) {
            Some(auth) => auth,
            None => {
                self.resolve_auth(api_key, optional_params, environment)
                    .await?
            }
        };
        Ok((auth.apply_preserving_existing(headers), auth))
    }

    fn auth_header_kind(&self) -> AuthHeaderKind {
        AuthHeaderKind::Bearer
    }

    fn requires_data_uri_document(&self) -> bool {
        false
    }

    fn response_handling(&self) -> OcrResponseHandling {
        OcrResponseHandling::Json
    }
}
