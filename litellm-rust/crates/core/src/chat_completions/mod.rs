//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`CoreClient::chat_completions`](crate::CoreClient::chat_completions) is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

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

impl crate::CoreClient {
    pub async fn chat_completions(
        &self,
        request: ChatCompletionsRequest<'_>,
    ) -> Result<ChatCompletionsResponse, Error> {
        self.chat_completions_with_hooks(request, &()).await
    }

    pub async fn chat_completions_with_hooks(
        &self,
        request: ChatCompletionsRequest<'_>,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<ChatCompletionsResponse, Error> {
        litellm_host::lifecycle::observe_unary(hooks.observer(), async {
            let http = self.provider_http()?;
            execute(&http, &self.resources().auth, request, hooks).await
        })
        .await
    }
}

async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    request: ChatCompletionsRequest<'_>,
    hooks: &impl litellm_host::hooks::RouteHooks<Error>,
) -> Result<ChatCompletionsResponse, Error> {
    let prepared = prepare_provider_request(resolve_request(request)?)?;
    let execute: futures_util::future::BoxFuture<'_, Result<ChatCompletionsResponse, Error>> =
        Box::pin(handler::execute(http, auth, prepared, hooks));
    execute.await
}
