use litellm_cache_response::CachePolicy;
use litellm_host::{
    interceptors::{ExecutionFacts, Interceptors, RawResponse},
    lifecycle::{CallEvent, ExecutionEvent},
    observation::ObservationSender,
};

use crate::{CallOptions, RouteError};

pub(crate) struct CallContext<'a, I> {
    pub interceptors: &'a I,
    pub observers: Option<ObservationSender>,
    pub cache: CachePolicy,
}

impl<'a, I: Interceptors<RouteError>> CallContext<'a, I> {
    pub fn new(interceptors: &'a I, options: CallOptions) -> Self {
        Self {
            interceptors,
            observers: options.observers,
            cache: options.cache.unwrap_or_default(),
        }
    }

    pub async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), RouteError> {
        if let Some(observers) = &self.observers {
            observers.emit(CallEvent::Execution(ExecutionEvent::ResultReady {
                facts: facts.clone(),
            }));
        }
        self.interceptors.result_ready(facts).await
    }

    pub async fn response_received(&self, body: &str) -> Result<(), RouteError> {
        let raw = RawResponse {
            body: body.to_owned(),
        };
        if let Some(observers) = &self.observers {
            observers.emit(CallEvent::Execution(
                ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
            ));
        }
        self.interceptors
            .after_provider_response(raw)
            .await
            .map_err(RouteError::post_call)
    }
}
