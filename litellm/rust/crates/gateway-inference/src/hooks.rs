//! The proxy's callback hooks around one call, at the points `ProxyBaseLLMRequestProcessing`
//! runs them. The [`Host`](crate::Host) the gateway was built with answers them, so a route
//! tells what happened to its call and never learns which callbacks hear it.

use std::{
    future::Future,
    sync::{Arc, OnceLock},
    time::SystemTime,
};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::{StreamExt, stream};
use litellm_framer::sse::SseBlock;
use litellm_gateway_auth::Authenticated;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_router_core::Deployment;
use serde_json::{Map, Value};

use crate::{
    Ended, Gateway, Host, record,
    serve::Endpoint,
    stream_event::Counts,
    stream_outcome::{self, Outcome},
};

/// The route a call came in on, each named as Python's `call_type` for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CallType {
    AnthropicMessages,
    Aocr,
}

/// A call a callback refused, as the proxy's `ProxyException` carries it to a route's handler.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Refusal {
    pub status: StatusCode,
    pub message: String,
    /// The exception's `type`, when it is not the one its status gives.
    pub kind: Option<String>,
    /// The exception's `code`, when it is not its status.
    pub code: Option<String>,
    /// In the order Python sends them.
    pub headers: Vec<(String, String)>,
}

impl Refusal {
    pub fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            ..Self::default()
        }
    }
}

/// How a hook ended a call: a callback refused it, or one failed on a hook where the proxy raises
/// what its hook raises. The failure keeps its cause, which `error.rs` renders.
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("{}", .0.message)]
    Refused(Refusal),
    #[error(transparent)]
    Failed(litellm_error::Source),
}

/// A call that was served, as the success log is told it.
#[derive(Clone, Debug, PartialEq)]
pub struct ServedCall {
    /// The model group that served it, an alias resolved, as the proxy logs it.
    pub model_group: String,
    /// The deployment that answered, `None` when the router no longer holds it.
    pub deployment: Option<Arc<Deployment>>,
    /// The provider's id for the answer, as the response or a stream's first event names it.
    pub response_id: Option<String>,
    /// As Python's litellm `Usage` counts it, `None` for an answer that carries no usage.
    pub usage: Option<ChatCompletionsUsage>,
    /// When a stream's first event arrived, `None` for a complete answer, which Python logs as
    /// starting its completion when it ended.
    pub first_event: Option<SystemTime>,
}

/// The hooks of one call, told in the order the proxy runs them:
///
/// - [`pre_call`](Self::pre_call) before the call is routed (`pre_call_hook`)
/// - [`post_call_success`](Self::post_call_success) on a complete answer only, as
///   `post_call_success_hook` never sees a stream
/// - [`response_headers`](Self::response_headers) on every served answer
///   (`post_call_response_headers_hook`)
/// - [`succeeded`](Self::succeeded) once the call is done: a complete answer after it goes out, a
///   stream when it ends (the success log)
/// - [`failed`](Self::failed) for a call that ends on an error, a refusal by a hook included
///   (`post_call_failure_hook`, then the failure log)
///
/// A stream the client drops before its end tells neither `succeeded` nor `failed`, so a callback
/// that holds something for the call gives it back when its state for the request is dropped.
/// Accepted difference: an attempt a retry or a fallback answered for is not logged as a
/// failure, since the route only hears how its call ended.
pub trait CallHooks: Send + Sync + 'static {
    /// Every callback sees the call's `body` in turn and may change it or refuse the call. The
    /// body comes back as the callbacks left it.
    fn pre_call(
        &self,
        body: Map<String, Value>,
    ) -> impl Future<Output = Result<Map<String, Value>, HookError>> + Send;

    fn post_call_success(&self) -> impl Future<Output = Result<(), HookError>> + Send;

    /// The headers the callbacks add to the answer.
    fn response_headers(&self) -> impl Future<Output = HeaderMap> + Send;

    fn succeeded(&self, served: ServedCall) -> impl Future<Output = ()> + Send;

    fn failed(&self) -> impl Future<Output = ()> + Send;
}

/// A served call's answer with what the success log is told once it goes out, `None` for a
/// stream, which tells its own end.
pub(crate) struct Answer {
    response: Response,
    complete: Option<ServedCall>,
}

/// What the hooks made of a served call before its answer is rendered.
pub(crate) struct Told {
    added: HeaderMap,
    complete: Option<ServedCall>,
}

impl Told {
    /// `response` with the headers the callbacks add.
    pub(crate) fn answer(self, response: impl IntoResponse) -> Answer {
        Answer {
            response: (self.added, response).into_response(),
            complete: self.complete,
        }
    }
}

/// Serves one call with the host's hooks around it: `serve` runs on the
/// body the pre-call hook left, a complete answer is logged off the request, as the proxy logs
/// it after it went out, and a call that ends on an error, a refusal by that hook included,
/// tells the hooks it failed.
pub(crate) async fn around<H, F, Fut>(
    gateway: &Gateway<H>,
    identity: &Authenticated<H::Rows>,
    body: Map<String, Value>,
    serve: F,
) -> Result<Response, Ended>
where
    H: Host,
    F: FnOnce(Arc<H::Hooks>, Map<String, Value>) -> Fut,
    Fut: Future<Output = Result<Answer, Ended>>,
{
    let hooks = Arc::new(gateway.host.hooks(identity));
    let served = match hooks.pre_call(body).await {
        Ok(body) => serve(hooks.clone(), body).await,
        Err(ended) => Err(ended.into()),
    };
    match served {
        Ok(Answer { response, complete }) => {
            if let Some(served) = complete {
                record::spawn_in_call(async move { hooks.succeeded(served).await });
            }
            Ok(response)
        }
        Err(ended) => {
            hooks.failed().await;
            Err(ended)
        }
    }
}

/// What served a call: the model group, an alias resolved, and the deployment that answered.
pub(crate) struct ServedBy {
    pub(crate) model_group: String,
    pub(crate) deployment: Option<Arc<Deployment>>,
}

/// What a stream's events told of its answer before it ended: when the first one arrived and the
/// id the provider gave the answer.
#[derive(Default)]
struct StreamSeen {
    first_event: OnceLock<SystemTime>,
    response_id: OnceLock<String>,
}

impl StreamSeen {
    fn see(&self, block: &SseBlock, response_id: fn(&str) -> Option<String>) {
        let _ = self.first_event.get_or_init(SystemTime::now);
        if self.response_id.get().is_none()
            && let Some(id) = block
                .event
                .as_ref()
                .and_then(|event| response_id(&event.data))
        {
            let _ = self.response_id.set(id);
        }
    }
}

/// Tells the hooks about the `outcome` that served a call of route `E`, which `served` says
/// served it. A complete answer tells what it counted, and a stream tells how it went once it
/// ends, through the same `stream_outcome::watched` the router hears it from.
pub(crate) async fn answered<E: Endpoint, K: CallHooks>(
    hooks: &Arc<K>,
    outcome: &mut E::Outcome,
    served: ServedBy,
) -> Result<Told, Ended> {
    let ServedBy {
        model_group,
        deployment,
    } = served;
    let Some((chunks, read)) = E::held(outcome) else {
        hooks.post_call_success().await?;
        return Ok(Told {
            added: hooks.response_headers().await,
            complete: Some(ServedCall {
                model_group,
                deployment,
                response_id: E::response_id(outcome),
                usage: E::counts(outcome).and_then(E::usage),
                first_event: None,
            }),
        });
    };
    let told = Told {
        added: hooks.response_headers().await,
        complete: None,
    };
    let hooks = hooks.clone();
    let usage: fn(Counts) -> Option<ChatCompletionsUsage> = E::usage;
    let response_id: fn(&str) -> Option<String> = E::stream_response_id;
    let seen = Arc::new(StreamSeen::default());
    let seeing = seen.clone();
    let unwatched = std::mem::replace(chunks, stream::empty().boxed())
        .inspect(move |block| {
            if let Ok(block) = block {
                seeing.see(block, response_id);
            }
        })
        .boxed();
    *chunks = stream_outcome::watched(
        unwatched,
        read,
        stream_outcome::broke,
        move |outcome: Outcome| {
            let (hooks, model_group, deployment, seen) = (
                hooks.clone(),
                model_group.clone(),
                deployment.clone(),
                seen.clone(),
            );
            async move {
                match outcome {
                    Ok(counts) => {
                        let served = ServedCall {
                            model_group,
                            deployment,
                            response_id: seen.response_id.get().cloned(),
                            usage: usage(counts),
                            first_event: seen.first_event.get().copied(),
                        };
                        hooks.succeeded(served).await;
                    }
                    Err(_) => hooks.failed().await,
                }
            }
        },
    );
    Ok(told)
}
