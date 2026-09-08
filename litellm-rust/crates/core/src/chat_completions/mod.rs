//! The `/chat/completions` call, the Rust equivalent of Python's
//! `litellm.completion()`.
//!
//! [`chat_completions`] is the top-level entrypoint: give it a model, the
//! OpenAI-shaped message list, the provider-mapped optional params, and
//! credentials, and it resolves the provider, translates the conversation,
//! calls the provider, and returns a typed OpenAI-shaped response.

use crate::Error;
mod common_utils;
pub mod conversation;
pub(crate) mod handler;
pub mod lifecycle;
pub mod request;
pub mod response_utils;
pub mod transformation;
pub mod types;

use serde_json::{Map, Value};

use handler::execute_chat_completions_provider_call;
pub use handler::{as_response_error, signed_headers, signed_headers_with_services};
use request::{parse_messages, resolve_provider_config, resolve_request};
use types::{ChatCompletionsRequest, ChatCompletionsResponse};

use crate::integrations::custom_logger::CallbackTiming;
use crate::integrations::types::Usage;
use crate::lifecycle::{
    CallLifecycleContext, ExecutedCall, RouteProjection, TerminalClassification, TerminalRecord,
};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn chat_completions(
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error> {
    chat_completions_with_services(
        crate::providers::auth::native_authorization_services(),
        request,
    )
    .await
}

pub async fn chat_completions_with_services<S>(
    services: &S,
    request: ChatCompletionsRequest<'_>,
) -> Result<ChatCompletionsResponse, Error>
where
    S: crate::providers::auth::AuthorizationServices,
{
    execute_chat_completions_provider_call(services, resolve_request(request)?).await
}

pub async fn chat_completions_with_terminal(
    request: ChatCompletionsRequest<'_>,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    with_terminal(chat_completions(request), context).await
}

pub async fn execute_settled_with_terminal(
    request: types::SettledChatRequest,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    with_terminal(handler::execute_settled_request(request), context).await
}

async fn with_terminal(
    call: impl std::future::Future<Output = Result<ChatCompletionsResponse, Error>>,
    context: CallLifecycleContext,
) -> ExecutedCall<ChatCompletionsResponse, Error> {
    let start_time = epoch_seconds();
    match call.await {
        Ok(response) => {
            let usage = Usage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
            };
            let projection = serde_json::to_value(&response).unwrap_or(Value::Null);
            let terminal = terminal(
                context,
                start_time,
                usage,
                TerminalClassification::Success,
                projection,
            );
            ExecutedCall::Success { response, terminal }
        }
        Err(error) => {
            let kind = match &error {
                Error::Auth(_) => "AuthError",
                Error::InvalidProvider(_) => "InvalidProvider",
                Error::InvalidRequest(_) => "InvalidRequest",
                Error::InvalidType { .. } => "InvalidType",
                Error::MissingField(_) => "MissingField",
                Error::Http { .. } => "HttpError",
                Error::InvalidResponse(_) => "InvalidResponse",
                Error::Network(_) => "NetworkError",
                Error::Connect(_) => "ConnectError",
                Error::Routing(_) => "RoutingError",
                Error::Unsupported(_) => "UnsupportedRequest",
            };
            let message = error.to_string();
            let terminal = terminal(
                context,
                start_time,
                Usage::default(),
                TerminalClassification::Failure {
                    kind: kind.into(),
                    message: message.clone(),
                },
                serde_json::json!({"kind": kind, "message": message}),
            );
            ExecutedCall::Failure { error, terminal }
        }
    }
}

fn terminal(
    context: CallLifecycleContext,
    start_time: f64,
    usage: Usage,
    classification: TerminalClassification,
    value: Value,
) -> TerminalRecord {
    TerminalRecord {
        call_id: context.litellm_call_id,
        trace_id: context.trace_id,
        attempt: context.attempt,
        call_type: context.call_type,
        model: context.model,
        provider: context.custom_llm_provider,
        timing: CallbackTiming::new(start_time, epoch_seconds()),
        usage,
        cost_inputs: Default::default(),
        classification,
        projection: RouteProjection::ChatCompletions { value },
    }
}

fn epoch_seconds() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
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
    let Ok((_, config)) = resolve_provider_config(model, custom_llm_provider) else {
        return Some("provider is not on the rust chat completions path");
    };
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
