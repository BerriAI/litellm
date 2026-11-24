//! What an attempt's provider call told, which the answer's headers name (`headers.rs`). The
//! route hands the core an [`Attempted`] as its interceptors, so it hears the request go out, the
//! provider answer and the result the core settled on. The route then puts it on its response.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::response::Response;
use litellm_host::interceptors::{
    Cost, ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
};
use litellm_inference::Error;
use parking_lot::Mutex;

#[derive(Clone, Debug)]
pub(crate) struct Attempted {
    /// The deployment's `litellm_params.model` (`x-litellm-model-name`).
    model: String,
    /// The deployment's `litellm_params.api_base`, as the config wrote it.
    api_base: Option<String>,
    heard: Arc<Mutex<Heard>>,
}

#[derive(Debug, Default)]
struct Heard {
    cache_key: Option<String>,
    cost: Option<Cost>,
    asked_at: Option<Instant>,
    /// How long the provider took to answer, to the head of a stream.
    provider: Option<Duration>,
    /// The headers of a provider response that streams. Python names the provider's headers on
    /// a stream only.
    stream_head: Option<Vec<(String, String)>>,
}

impl Attempted {
    /// An attempt on the deployment whose `litellm_params` name `model` and `api_base`.
    pub(crate) fn new(model: String, api_base: Option<String>) -> Self {
        Self {
            model,
            api_base,
            heard: Arc::default(),
        }
    }

    /// Puts what the attempt told on the response it served, for [`crate::headers::told`].
    pub(crate) fn told(self, mut response: Response) -> Response {
        response.extensions_mut().insert(self);
        response
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn api_base(&self) -> Option<&str> {
        self.api_base.as_deref()
    }

    pub(crate) fn cache_key(&self) -> Option<String> {
        self.heard.lock().cache_key.clone()
    }

    pub(crate) fn cost(&self) -> Option<Cost> {
        self.heard.lock().cost.clone()
    }

    pub(crate) fn provider(&self) -> Option<Duration> {
        self.heard.lock().provider
    }

    pub(crate) fn stream_head(&self) -> Option<Vec<(String, String)>> {
        self.heard.lock().stream_head.clone()
    }
}

impl Interceptors<Error> for Attempted {
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        _: RequestContext,
    ) -> Result<WireRequest, Error> {
        self.heard.lock().asked_at = Some(Instant::now());
        Ok(wire)
    }

    async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
        let mut heard = self.heard.lock();
        heard.provider = heard.asked_at.map(|asked_at| asked_at.elapsed());
        heard.stream_head = raw
            .head
            .filter(|head| head.streamed)
            .map(|head| head.headers);
        Ok(())
    }

    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        let mut heard = self.heard.lock();
        if let ResultSource::Cache { key } = facts.source {
            heard.cache_key = Some(key);
        }
        heard.cost = Some(facts.cost);
        Ok(())
    }
}
