use serde_json::Value;

use crate::error::CoreResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingsAuthStrategy {
    Bearer,
    Header(&'static str),
}

impl EmbeddingsAuthStrategy {
    pub fn header_name(self) -> &'static str {
        match self {
            Self::Bearer => "authorization",
            Self::Header(header_name) => header_name,
        }
    }
}

pub trait EmbeddingsProviderConfig: Sync {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn auth_strategy(&self) -> EmbeddingsAuthStrategy {
        EmbeddingsAuthStrategy::Bearer
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn transform_request(&self, request: Value) -> CoreResult<Value> {
        Ok(request)
    }

    fn transform_response(&self, _model: &str, response: Value) -> CoreResult<Value> {
        Ok(response)
    }
}
