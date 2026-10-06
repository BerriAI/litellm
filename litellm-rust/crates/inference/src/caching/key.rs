use litellm_cache_response::{CacheKeyInput, CacheTarget, extra_headers};
use serde_json::Value;

use super::Cachable;
use crate::RouteError;

pub trait CacheKeyProjection {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError>;
}

impl CacheKeyProjection for crate::chat_completions::types::ChatCompletionsRequest<'_> {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        Ok(CacheKeyInput::forwarded(
            <crate::chat_completions::route::ChatCompletions as Cachable>::SURFACE,
            CacheTarget::resolve(
                model_group,
                self.model,
                self.custom_llm_provider,
                self.api_base,
            ),
            self.optional_params
                .clone()
                .into_iter()
                .chain([("messages".into(), self.messages.clone())]),
            extra_headers(self.extra_headers.as_ref()),
        ))
    }
}

impl CacheKeyProjection for crate::responses::types::ResponsesCall {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        Ok(CacheKeyInput::forwarded(
            <crate::responses::route::Responses as Cachable>::SURFACE,
            CacheTarget::resolve(
                model_group,
                &self.model,
                self.custom_llm_provider.as_deref(),
                self.api_base.as_deref(),
            ),
            self.optional_params
                .clone()
                .into_iter()
                .chain([("input".into(), self.input.clone())]),
            extra_headers(self.extra_headers.as_ref()),
        ))
    }
}

impl CacheKeyProjection for crate::messages::MessagesCall {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError> {
        let invalid =
            |error: serde_json::Error| RouteError::InvalidRequest(error.to_string().into());
        let Value::Object(body) = serde_json::to_value(&self.body).map_err(invalid)? else {
            return Err(RouteError::InvalidRequest(
                "messages body must be an object".into(),
            ));
        };
        let provider_specific_header = self
            .provider_specific_header
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(invalid)?
            .map(|header| ("provider_specific_header", header));
        Ok(CacheKeyInput::forwarded(
            <crate::messages::route::Messages as Cachable>::SURFACE,
            CacheTarget::resolve(
                model_group,
                &self.body.model,
                self.custom_llm_provider.as_deref(),
                self.api_base.as_deref(),
            ),
            body,
            extra_headers(self.extra_headers.as_ref())
                .into_iter()
                .chain(provider_specific_header),
        ))
    }
}
