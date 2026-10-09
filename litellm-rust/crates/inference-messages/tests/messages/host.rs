use litellm_host::lifecycle::ExecutionEvent;
use std::sync::Mutex;

use litellm_host::{
    interceptors::{ExecutionFacts, RequestContext, ResultSource, WireRequest},
    lifecycle::CallEvent,
};
use litellm_inference_messages::{MessagesCallResponse, route::Messages};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use litellm_llms::base_llm::messages::context::MessagesModelCapabilities as AnthropicModelCapabilities;
use rstest::rstest;

use super::*;

type Rewrite =
    Box<dyn Fn(WireRequest) -> Result<WireRequest, litellm_host::error::HookError> + Send + Sync>;

/// Projects like `LocalMessagesHost`, answers `before_provider_request` through `rewrite`, and keeps
/// every event the driver emits.
struct RecordingHost {
    call: LocalMessagesHost,
    rewrite: Rewrite,
    events: super::support::Observations,
    optional_params: Mutex<Vec<Value>>,
    facts: Mutex<Vec<ExecutionFacts>>,
    reject_result: bool,
}

impl RecordingHost {
    fn new(call: MessagesCall, rewrite: Rewrite) -> Self {
        Self {
            call: LocalMessagesHost::new(call),
            rewrite,
            events: super::support::Observations::default(),
            optional_params: Mutex::new(Vec::new()),
            facts: Mutex::new(Vec::new()),
            reject_result: false,
        }
    }

    fn passthrough(call: MessagesCall) -> Self {
        Self::new(call, Box::new(Ok))
    }

    fn raw_responses(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                    Some(raw.body.clone())
                }
                _ => None,
            })
            .collect()
    }
}

impl RecordingHost {
    pub fn request(&self) -> Result<MessagesCall, Error> {
        self.call.request()
    }
    pub fn runtime(&self) -> litellm_host_native::in_process::Host<'_, (), Self, ()> {
        litellm_host_native::in_process::Host {
            services: &(),
            hooks: self,
            stream: &(),
        }
    }
}

impl litellm_host::hooks::NativeHooks for RecordingHost {
    fn before_provider_request(
        &self,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, litellm_host::error::HookError> {
        self.optional_params
            .lock()
            .unwrap()
            .push(context.optional_params.clone());
        (self.rewrite)(*wire).map(Box::new)
    }

    fn result_ready(&self, facts: &ExecutionFacts) -> Result<(), litellm_host::error::HookError> {
        self.facts.lock().unwrap().push(facts.clone());
        if self.reject_result {
            return Err(litellm_host::error::HookError::Rejected {
                reason: "result rejected".into(),
            });
        }
        Ok(())
    }

    fn on_event(&self, event: &litellm_host::lifecycle::CallEvent) {
        self.events.sender.emit(event.clone());
    }
}

/// The same hooks for the direct path, which only ever sees the awaited boundaries.
impl litellm_host::interceptors::Interceptors<<Messages as litellm_host::protocol::Protocol>::Error>
    for RecordingHost
{
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        self.facts.lock().unwrap().push(facts);
        if self.reject_result {
            return Err(Error::Unsupported("result rejected"));
        }
        Ok(())
    }

    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        self.optional_params
            .lock()
            .unwrap()
            .push(context.optional_params.clone());
        (self.rewrite)(wire).map_err(Error::from)
    }
    async fn after_provider_response(
        &self,
        raw: litellm_host::interceptors::RawResponse,
    ) -> Result<(), <Messages as litellm_host::protocol::Protocol>::Error> {
        self.events
            .sender
            .emit(litellm_host::lifecycle::CallEvent::Execution(
                litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
            ));
        Ok(())
    }
}

#[rstest]
#[case::native_unary(false, false)]
#[case::native_stream(true, false)]
#[case::hosted_unary(false, true)]
#[case::hosted_stream(true, true)]
#[tokio::test]
async fn rejected_results_are_not_delivered_or_cached(
    call: MessagesCall,
    #[case] streaming: bool,
    #[case] hosted: bool,
) {
    use futures_util::TryStreamExt;
    use litellm_cache_memory::InMemoryCache;
    use litellm_cache_response::{CacheScope, ResponseCache, ScopedCache};

    let response = if streaming {
        ResponseTemplate::new(200).set_body_raw(
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            "text/event-stream",
        )
    } else {
        message_response()
    };
    let upstream = upstream([response.clone(), response]).await;
    let route = messages_route(no_secrets()).with_cache(ScopedCache::new(
        Arc::new(ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))),
        CacheScope::Shared,
    ));
    for (reject, expected_requests, cached) in [
        (true, 1, false),
        (false, 2, false),
        (true, 2, true),
        (false, 2, true),
    ] {
        let request = authenticated(
            with_fields(
                MessagesCall {
                    body: call.body.clone(),
                    ..super::call()
                },
                json!({"stream": streaming}),
            ),
            upstream.uri(),
        );
        let host = RecordingHost {
            reject_result: reject,
            ..RecordingHost::passthrough(request)
        };
        let result = if hosted {
            litellm_host_native::in_process::run_hosted(
                route.clone().machine(host.request().unwrap(), None),
                host.runtime(),
            )
            .await
            .map(|_| ())
        } else {
            match route.execute(host.request().unwrap(), &host, None).await {
                Ok(MessagesCallResponse::Complete(_)) => Ok(()),
                Ok(MessagesCallResponse::Stream { chunks, .. }) => {
                    chunks.try_collect::<Vec<_>>().await.map(|_| ())
                }
                Err(error) => Err(error),
            }
        };
        let expected = match (reject, hosted) {
            (false, _) => Ok(()),
            (true, true) => Err(Error::Hook(litellm_host::error::HookError::Rejected {
                reason: "result rejected".into(),
            })),
            (true, false) => Err(Error::Unsupported("result rejected")),
        };
        assert_eq!(result, expected);
        assert_eq!(received(&upstream).await.len(), expected_requests);
        let facts = host.facts.lock().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(
            matches!(facts[0].source, ResultSource::Cache { .. }),
            cached
        );
    }
}

async fn run_through(host: &RecordingHost) -> Result<MessagesOutput, Error> {
    litellm_host_native::in_process::run_hosted(
        machine(Arc::new(RecordingSecrets::empty()))(host.request()?),
        host.runtime(),
    )
    .await
}

fn authenticated(call: MessagesCall, api_base: String) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        ..call
    }
}

#[rstest]
#[tokio::test]
async fn what_before_send_returns_is_what_the_provider_receives(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::new(
        authenticated(call, upstream.uri()),
        Box::new(|wire| {
            let mut body = wire.body;
            body["system"] = json!("added by the host");
            Ok(WireRequest {
                headers: wire
                    .headers
                    .into_iter()
                    .chain([("x-host".to_string(), "seen".to_string())])
                    .collect(),
                body,
                ..wire
            })
        }),
    );

    run_through(&host).await.expect("messages call succeeds");

    let request = only_request(&upstream).await;
    assert_eq!(request.json()["system"], "added by the host");
    assert_eq!(request.header("x-host"), Some("seen"));
    assert_eq!(request.header("x-api-key"), Some("sk-ant"));
}

#[rstest]
#[case::enable(false, json!(true), Some(true))]
#[case::disable(true, json!(false), Some(false))]
#[case::null(true, Value::Null, Some(false))]
#[case::invalid(false, json!("true"), None)]
#[tokio::test]
async fn response_mode_follows_the_intercepted_request(
    call: MessagesCall,
    traces: TraceCapture,
    #[case] original_stream: bool,
    #[case] rewritten_stream: Value,
    #[case] expected_stream: Option<bool>,
) {
    use futures_util::TryStreamExt;

    let sse = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let response = if expected_stream == Some(true) {
        ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
    } else {
        message_response()
    };
    let upstream = upstream([response]).await;
    let rewrite = rewritten_stream.clone();
    let host = RecordingHost::new(
        authenticated(
            with_fields(call, json!({"stream": original_stream})),
            upstream.uri(),
        ),
        Box::new(move |wire| {
            let mut body = wire.body;
            body["stream"] = rewrite.clone();
            Ok(WireRequest { body, ..wire })
        }),
    );
    let result = traces
        .logger()
        .instrument(async {
            let output = messages_route(no_secrets())
                .execute(host.request()?, &host, None)
                .await?;
            match output {
                MessagesCallResponse::Stream { chunks, .. } => {
                    assert_eq!(expected_stream, Some(true));
                    assert_eq!(
                        chunks.try_collect::<Vec<_>>().await?.concat(),
                        sse.as_bytes()
                    );
                }
                MessagesCallResponse::Complete(message) => {
                    assert_eq!(expected_stream, Some(false));
                    assert_eq!(*message, serde_json::from_value(message_body()).unwrap());
                }
            }
            Ok::<_, Error>(())
        })
        .await;

    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    let Some(expected_stream) = expected_stream else {
        assert!(matches!(result, Err(Error::InvalidRequest(_))));
        assert!(received(&upstream).await.is_empty());
        assert_eq!(summaries[0]["outcome"], "failure");
        return;
    };
    result.unwrap();
    assert_eq!(
        only_request(&upstream).await.json()["stream"],
        rewritten_stream
    );
    assert_eq!(host.raw_responses().len(), usize::from(!expected_stream));
    assert_eq!(summaries[0]["stream"], expected_stream);
    assert_eq!(summaries[0]["outcome"], "success");
}

#[rstest]
#[tokio::test]
async fn a_before_send_failure_never_sends(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::new(
        authenticated(call, upstream.uri()),
        Box::new(|_| {
            Err(litellm_host::error::HookError::Rejected {
                reason: "vetoed by the host".into(),
            })
        }),
    );

    let error = run_through(&host)
        .await
        .expect_err("the host failure fails the call");

    assert_eq!(
        error,
        Error::Hook(litellm_host::error::HookError::Rejected {
            reason: "vetoed by the host".into(),
        })
    );
    assert!(received(&upstream).await.is_empty());
    assert!(host.raw_responses().is_empty());
}

#[rstest]
#[tokio::test]
async fn the_raw_upstream_text_is_emitted_once_for_a_message(call: MessagesCall) {
    let raw = message_body();
    let upstream = upstream([json_response(raw.clone())]).await;
    let host = RecordingHost::passthrough(authenticated(call, upstream.uri()));

    let output = run_through(&host).await.expect("messages call succeeds");

    assert!(matches!(output, MessagesOutput::Complete(_)));
    let [emitted] = <[String; 1]>::try_from(host.raw_responses())
        .unwrap_or_else(|raws| panic!("expected one raw response, got {}", raws.len()));
    assert_eq!(serde_json::from_str::<Value>(&emitted).unwrap(), raw);
}

#[rstest]
#[case::upstream_error(ResponseTemplate::new(500).set_body_string("boom"))]
#[case::stream(ResponseTemplate::new(200).set_body_raw("event: message_stop\ndata: {}\n\n", "text/event-stream"))]
#[tokio::test]
async fn no_raw_response_is_emitted_for_a_stream_or_a_failure(
    call: MessagesCall,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;
    let host = RecordingHost::passthrough(authenticated(
        with_fields(call, json!({"stream": true})),
        upstream.uri(),
    ));

    let _ = run_through(&host).await;

    assert_eq!(received(&upstream).await.len(), 1);
    assert!(host.raw_responses().is_empty());
}

/// Python logs `optional_params` as what it is about to send, so a dropped param must
/// not resurface in callbacks.
#[rstest]
#[tokio::test]
async fn the_request_context_carries_the_shaped_params_without_model_or_messages(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::passthrough(authenticated(
        MessagesCall {
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    drop_params: true,
                    ..MessagesSettings::default()
                },
                capabilities: AnthropicModelCapabilities {
                    supports_sampling_params: false,
                    ..AnthropicModelCapabilities::default()
                },
            },
            ..with_fields(call, json!({"temperature": 0.2}))
        },
        upstream.uri(),
    ));

    run_through(&host).await.expect("messages call succeeds");

    let [optional_params] = <[Value; 1]>::try_from(host.optional_params.into_inner().unwrap())
        .unwrap_or_else(|seen| panic!("before_provider_request runs once, saw {}", seen.len()));
    assert_eq!(optional_params, json!({"max_tokens": 16}));
}
