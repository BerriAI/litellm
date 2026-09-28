pub mod route;
pub mod types;
pub use crate::error::RouteError as Error;
mod common_utils;
pub(crate) mod handler;
mod prepare;
use litellm_types::utils::ChatCompletionsResponse;
use prepare::{parse_messages, prepare_provider_request, resolve_provider_config, resolve_request};
use serde_json::{Map, Value};

use crate::chat_completions::types::ChatCompletionsRequest;
use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

#[derive(Clone)]
pub struct ChatCompletionsRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

/// Whether the core would accept this request, without resolving credentials or
/// touching the network.
///
/// A host that keeps the Python implementation asks this first so it can emit
/// its pre-call logging exactly once, on whichever path is about to run.
/// Returns the decline reason, or `None` when the request is accepted.
pub fn chat_completions_decline_reason(
    model: &str,
    custom_llm_provider: Option<&str>,
    messages: Value,
    optional_params: &Map<String, Value>,
) -> Option<&'static str> {
    let Ok(resolved) = resolve_provider_config(model, custom_llm_provider) else {
        return Some("provider is not on the rust chat completions path");
    };
    let config = resolved.config;
    let Ok(messages) = parse_messages(messages) else {
        return Some("unreadable message list");
    };
    if messages.is_empty() {
        return Some("empty message list");
    }
    config
        .unsupported_reason(&messages, optional_params)
        .map(|reason| reason.0)
}

impl ChatCompletionsRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    pub async fn execute(
        &self,
        request: ChatCompletionsRequest<'_>,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<ChatCompletionsResponse, Error> {
        litellm_host::lifecycle::observe_unary(hooks.observer(), self.run(request, hooks)).await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %request.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(
        &self,
        request: ChatCompletionsRequest<'_>,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<ChatCompletionsResponse, Error> {
        crate::diagnostic::unary(async {
            let resolved = resolve_request(request)?;
            let snapshot = self
                .secrets
                .resolve(&resolved.config.secret_names())
                .await?;
            let prepared = prepare_provider_request(resolved, snapshot)?;
            crate::diagnostic::provider(&prepared.model, &prepared.custom_llm_provider);
            let execute: futures_util::future::BoxFuture<
                '_,
                Result<ChatCompletionsResponse, Error>,
            > = Box::pin(handler::execute(&self.http, &self.auth, prepared, hooks));
            execute.await
        })
        .await
    }
}
