//! The router: construction, deployment lookup, and route methods.

use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::CoreResult;

use crate::deployment::Deployment;
use crate::strategy::RoutingStrategy;

/// Load-balancing router over a `model_list`.
#[derive(Clone, Debug, Default)]
pub struct Router {
    model_list: Vec<Deployment>,
    routing_strategy: RoutingStrategy,
}

impl Router {
    /// Build a router from a `model_list` using the default `simple-shuffle` strategy.
    pub fn new(model_list: Vec<Deployment>) -> Self {
        Self {
            model_list,
            routing_strategy: RoutingStrategy::SimpleShuffle,
        }
    }

    /// Pick a deployment for `model` per the routing strategy. Returns `None`
    /// when no deployment is registered under that `model_name`.
    pub fn get_available_deployment(&self, model: &str) -> Option<&Deployment> {
        let candidates: Vec<&Deployment> = self
            .model_list
            .iter()
            .filter(|deployment| deployment.model_name == model)
            .collect();
        self.routing_strategy.select(&candidates)
    }

    /// Select a deployment for `model` and invoke the realtime route against it.
    ///
    /// Mirrors `router._arealtime`: resolve a deployment, then call the
    /// underlying provider entry point with that deployment's params.
    pub async fn realtime(
        &self,
        model: &str,
        input_events: Vec<RealtimeEvent>,
        timeout: Option<Duration>,
    ) -> CoreResult<Vec<RealtimeEvent>> {
        let deployment = self.get_available_deployment(model).ok_or_else(|| {
            CoreError::Routing(format!("no deployment available for model '{model}'"))
        })?;
        let params = &deployment.litellm_params;
        // Strip a leading `openai/` so the OpenAI-only realtime fn gets the bare model.
        let provider_model = params
            .model
            .strip_prefix("openai/")
            .unwrap_or(&params.model);

        litellm_providers::realtime::realtime(
            provider_model,
            input_events,
            params.api_key.as_deref(),
            params.api_base.as_deref(),
            timeout,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::LiteLLMParams;

    fn deployment(name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
        }
    }

    #[test]
    fn selects_a_matching_deployment() {
        let router = Router::new(vec![
            deployment("gpt-realtime", "gpt-realtime"),
            deployment("other", "other-model"),
        ]);
        let chosen = router
            .get_available_deployment("gpt-realtime")
            .expect("a deployment should match");
        assert_eq!(chosen.model_name, "gpt-realtime");
    }

    #[test]
    fn unknown_model_returns_none() {
        let router = Router::new(vec![deployment("gpt-realtime", "gpt-realtime")]);
        assert!(router.get_available_deployment("missing").is_none());
    }

    #[tokio::test]
    async fn realtime_errors_for_unknown_model() {
        let router = Router::new(vec![deployment("gpt-realtime", "gpt-realtime")]);
        let err = router
            .realtime("missing", Vec::new(), None)
            .await
            .expect_err("unknown model should error");
        assert!(matches!(err, CoreError::Routing(_)));
    }
}
