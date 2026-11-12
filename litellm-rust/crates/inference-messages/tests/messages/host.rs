use litellm_core_utils::decline::Decline;
use litellm_host::lifecycle::ExecutionEvent;
use std::sync::Mutex;

use litellm_host::{
    interceptors::{ExecutionFacts, RawResponse, RequestContext, ResultSource, WireRequest},
    lifecycle::CallEvent,
};
use litellm_inference_messages::{MessagesBody, MessagesCallResponse, route::Messages};
use litellm_inference_testing::{RecordingSecrets, no_secrets};
use litellm_llms::base_llm::messages::context::{
    MessagesModelCapabilities as AnthropicModelCapabilities, SupportedEffortTiers,
};
use rstest::rstest;

use super::*;

type Rewrite = Box<dyn Fn(WireRequest) -> Result<WireRequest, Error> + Send + Sync>;

/// Projects like `LocalMessagesHost`, answers `before_provider_request` through `rewrite`, and keeps
/// every event the driver emits.
struct RecordingHost {
    call: LocalMessagesHost,
    rewrite: Rewrite,
    events: super::support::Observations,
    optional_params: Mutex<Vec<Value>>,
    facts: Mutex<Vec<ExecutionFacts>>,
    reject_result: bool,
    reject_response: bool,
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
            reject_response: false,
        }
    }

    fn passthrough(call: MessagesCall) -> Self {
        Self::new(call, Box::new(Ok))
    }

    fn raw_heads(&self) -> Vec<RawResponse> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                    Some(raw.clone())
                }
                _ => None,
            })
            .collect()
    }

    fn raw_responses(&self) -> Vec<String> {
        self.raw_heads().into_iter().map(|raw| raw.body).collect()
    }
}

impl RecordingHost {
    pub fn request(&self) -> Result<MessagesCall, Error> {
        self.call.request()
    }
    pub fn runtime(&self) -> litellm_host_native::in_process::Host<'_, (), Self, ()> {
        litellm_host_native::in_process::Host {
            services: &(),
            interceptors: self,
            stream: &(),
            observers: Some(&self.events.sender),
        }
    }
}

impl litellm_host::lifecycle::CallObserver for RecordingHost {
    fn observe(&self, event: litellm_host::lifecycle::CallEvent) {
        self.events.sender.emit(event);
    }
}
impl litellm_host::interceptors::Interceptors<<Messages as litellm_host::protocol::Protocol>::Error>
    for RecordingHost
{
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        self.facts.lock().unwrap().push(facts);
        if self.reject_result {
            return Err(Error::Declined(Decline::new("result rejected")));
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
        (self.rewrite)(wire)
    }
    async fn after_provider_response(
        &self,
        raw: litellm_host::interceptors::RawResponse,
    ) -> Result<(), <Messages as litellm_host::protocol::Protocol>::Error> {
        if self.reject_response {
            return Err(Error::Declined(Decline::new("response rejected")));
        }
        litellm_host::lifecycle::CallObserver::observe(
            self,
            litellm_host::lifecycle::CallEvent::Execution(
                litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
            ),
        );
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
    #[values(false, true)] cached: bool,
) {
    use futures_util::TryStreamExt;
    use litellm_cache_memory::InMemoryCache;
    use litellm_cache_response::{CacheScope, ResponseCache, ScopedCache};

    let response = if streaming {
        ResponseTemplate::new(200).set_body_raw(
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
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
    let request = || {
        authenticated(
            with_fields(
                MessagesCall {
                    body: call.body.clone(),
                    ..super::call()
                },
                json!({"stream": streaming}),
            ),
            upstream.uri(),
        )
    };
    let host = RecordingHost {
        reject_result: true,
        ..RecordingHost::passthrough(request())
    };
    if cached {
        let output = route.execute(request(), &(), None).await.unwrap();
        if let MessagesCallResponse::Stream { chunks, .. } = output {
            chunks.try_collect::<Vec<_>>().await.unwrap();
        }
    }
    let result = if hosted {
        litellm_host_native::in_process::run_hosted(
            route.clone().machine(host.request().unwrap(), None),
            host.runtime(),
        )
        .await
        .map(|_| ())
    } else {
        route
            .execute(host.request().unwrap(), &host, None)
            .await
            .map(|_| ())
    };
    let error = result.unwrap_err();
    assert_eq!(error.declined().is_some(), cached);
    assert_eq!(error.dispatched(), !cached);
    assert!(
        litellm_error::Report::new(&error)
            .to_string()
            .contains("result rejected")
    );
    assert_eq!(received(&upstream).await.len(), 1);
    {
        let facts = host.facts.lock().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(
            matches!(facts[0].source, ResultSource::Cache { .. }),
            cached
        );
    }
    let output = route.execute(request(), &(), None).await.unwrap();
    if let MessagesCallResponse::Stream { chunks, .. } = output {
        chunks.try_collect::<Vec<_>>().await.unwrap();
    }
    assert_eq!(received(&upstream).await.len(), if cached { 1 } else { 2 });
}

#[rstest]
#[case::native(false)]
#[case::hosted(true)]
#[tokio::test]
async fn response_hook_declines_are_terminal(call: MessagesCall, #[case] hosted: bool) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost {
        reject_response: true,
        ..RecordingHost::passthrough(authenticated(call, upstream.uri()))
    };
    let route = messages_route(no_secrets());
    let result = if hosted {
        litellm_host_native::in_process::run_hosted(
            route.machine(host.request().unwrap(), None),
            host.runtime(),
        )
        .await
        .map(|_| ())
    } else {
        route
            .execute(host.request().unwrap(), &host, None)
            .await
            .map(|_| ())
    };
    let error = result.unwrap_err();
    assert!(error.declined().is_none());
    assert!(error.dispatched());
    assert!(
        litellm_error::Report::new(&error)
            .to_string()
            .contains("response rejected")
    );
    assert_eq!(received(&upstream).await.len(), 1);
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
#[case::enable(false, json!(true), true)]
#[case::disable(true, json!(false), false)]
#[case::null(true, Value::Null, false)]
#[case::truthy_string(false, json!("false"), true)]
#[case::falsy_integer(true, json!(0), false)]
#[tokio::test]
async fn response_mode_follows_the_intercepted_request(
    call: MessagesCall,
    traces: TraceCapture,
    #[case] original_stream: bool,
    #[case] rewritten_stream: Value,
    #[case] expected_stream: bool,
) {
    use futures_util::TryStreamExt;

    let sse = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let response = if expected_stream {
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
                    assert!(expected_stream);
                    assert_eq!(
                        chunks.try_collect::<Vec<_>>().await?.concat(),
                        sse.as_bytes()
                    );
                }
                MessagesCallResponse::Complete(message) => {
                    assert!(!expected_stream);
                    assert_eq!(
                        message,
                        MessagesBody::Message(serde_json::from_value(message_body()).unwrap())
                    );
                }
            }
            Ok::<_, Error>(())
        })
        .await;

    let summaries = traces.summaries("litellm.route");
    assert_eq!(summaries.len(), 1);
    result.unwrap();
    assert_eq!(
        only_request(&upstream).await.json()["stream"],
        rewritten_stream
    );
    assert_eq!(host.raw_responses().len(), 1);
    assert_eq!(summaries[0]["stream"], expected_stream);
    assert_eq!(summaries[0]["outcome"], "success");
}

#[rstest]
#[tokio::test]
async fn a_before_send_failure_never_sends(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::new(
        authenticated(call, upstream.uri()),
        Box::new(|_| Err(litellm_llms::Error::MissingField("vetoed by the host").into())),
    );

    let error = run_through(&host)
        .await
        .expect_err("the host failure fails the call");

    assert_eq!(
        error,
        Error::from(litellm_llms::Error::MissingField("vetoed by the host"))
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
#[tokio::test]
async fn no_raw_response_is_emitted_for_a_failure(call: MessagesCall) {
    let upstream = upstream([ResponseTemplate::new(500).set_body_string("boom")]).await;
    let host = RecordingHost::passthrough(authenticated(
        with_fields(call, json!({"stream": true})),
        upstream.uri(),
    ));

    let _ = run_through(&host).await;

    assert_eq!(received(&upstream).await.len(), 1);
    assert!(host.raw_responses().is_empty());
}

#[rstest]
#[tokio::test]
async fn a_stream_emits_its_head_without_a_body_before_any_chunk(call: MessagesCall) {
    let upstream = upstream([ResponseTemplate::new(200)
        .insert_header("request-id", "req_stream")
        .set_body_raw("event: message_stop\ndata: {}\n\n", "text/event-stream")])
    .await;
    let host = RecordingHost::passthrough(authenticated(
        with_fields(call, json!({"stream": true})),
        upstream.uri(),
    ));

    let _ = run_through(&host).await;

    let [raw] = <[RawResponse; 1]>::try_from(host.raw_heads())
        .unwrap_or_else(|raws| panic!("expected one raw response, got {}", raws.len()));
    assert_eq!(raw.body, "");
    let head = raw.head.expect("the stream's head is forwarded");
    assert_eq!(head.status, 200);
    assert!(
        head.headers
            .iter()
            .any(|(name, value)| name == "request-id" && value == "req_stream")
    );
}

/// Python logs `optional_params` after its requested-param gate, so a dropped param must
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

#[rstest]
#[tokio::test]
async fn the_request_context_keeps_the_nested_fields_additional_drop_params_strip(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;
    let tool = json!({"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]});
    let host = RecordingHost::passthrough(authenticated(
        MessagesCall {
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    additional_drop_params: vec!["tools[*].input_examples".into()],
                    ..MessagesSettings::default()
                },
                ..MessagesShaping::default()
            },
            ..with_fields(call, json!({"tools": [tool.clone()]}))
        },
        upstream.uri(),
    ));

    run_through(&host).await.expect("messages call succeeds");

    let [optional_params] = <[Value; 1]>::try_from(host.optional_params.into_inner().unwrap())
        .unwrap_or_else(|seen| panic!("before_provider_request runs once, saw {}", seen.len()));
    assert_eq!(optional_params["tools"], json!([tool]));
    assert_eq!(
        only_request(&upstream).await.json()["tools"][0].get("input_examples"),
        None
    );
}

#[rstest]
#[tokio::test]
async fn the_request_context_leaves_out_a_listed_param_the_model_refuses(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::passthrough(authenticated(
        MessagesCall {
            shaping: MessagesShaping {
                settings: MessagesSettings {
                    additional_drop_params: vec!["temperature".into()],
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

#[rstest]
#[case::reasoning_effort_on_an_adaptive_model(
    AnthropicModelCapabilities {
        supports_reasoning: true,
        supports_adaptive_thinking: true,
        supports_output_config: true,
        effort_tiers: SupportedEffortTiers { low: true, ..SupportedEffortTiers::default() },
        ..AnthropicModelCapabilities::default()
    },
    json!({"reasoning_effort": "low"})
)]
#[case::adaptive_payload_on_a_legacy_model(
    AnthropicModelCapabilities {
        supports_reasoning: true,
        ..AnthropicModelCapabilities::default()
    },
    json!({"max_tokens": 3000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}})
)]
#[tokio::test]
async fn the_request_context_carries_the_requested_reasoning_not_its_translation(
    call: MessagesCall,
    #[case] capabilities: AnthropicModelCapabilities,
    #[case] fields: Value,
) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::passthrough(authenticated(
        MessagesCall {
            shaping: MessagesShaping {
                capabilities,
                ..MessagesShaping::default()
            },
            ..with_fields(call, fields.clone())
        },
        upstream.uri(),
    ));

    run_through(&host).await.expect("messages call succeeds");

    let [optional_params] = <[Value; 1]>::try_from(host.optional_params.into_inner().unwrap())
        .unwrap_or_else(|seen| panic!("before_provider_request runs once, saw {}", seen.len()));
    assert_eq!(
        optional_params,
        Value::Object(
            object(json!({"max_tokens": 16}))
                .into_iter()
                .chain(object(fields))
                .collect()
        )
    );
}

#[derive(Default)]
struct RecordingAnswer(Mutex<Vec<bool>>);

impl litellm_inference_messages::HostAnswer for RecordingAnswer {
    fn answer(&self, stream: bool) -> futures_util::future::BoxFuture<'_, Result<(), Error>> {
        self.0.lock().unwrap().push(stream);
        Box::pin(async { Ok(()) })
    }
}

#[rstest]
#[case::unary(false)]
#[case::stream(true)]
#[tokio::test]
async fn an_intercepted_call_is_answered_by_the_host_before_the_mock_and_the_provider(
    call: MessagesCall,
    #[case] streaming: bool,
) {
    let upstream = upstream([message_response()]).await;
    let answer = Arc::new(RecordingAnswer::default());
    let route = messages_route(no_secrets()).with_host_answer(answer.clone());
    let mut request = authenticated(
        with_fields(call, json!({"stream": streaming})),
        upstream.uri(),
    );
    request.shaping.settings.intercepted = true;
    request.shaping.settings.mock_response = Some(object(json!({"id": "mocked"})));
    let host = RecordingHost::passthrough(request);

    let output = route
        .execute(host.request().unwrap(), &host, None)
        .await
        .unwrap();

    assert!(matches!(
        output,
        MessagesCallResponse::Complete(MessagesBody::Unchecked(body)) if body.is_empty()
    ));
    assert_eq!(*answer.0.lock().unwrap(), vec![streaming]);
    assert!(received(&upstream).await.is_empty());
    let facts = host.facts.lock().unwrap();
    assert_eq!(facts.len(), 1);
    assert_eq!(facts[0].source, ResultSource::Host);
}

#[rstest]
#[tokio::test]
async fn a_call_no_interceptor_owns_never_asks_the_host(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let answer = Arc::new(RecordingAnswer::default());
    let route = messages_route(no_secrets()).with_host_answer(answer.clone());
    let host = RecordingHost::passthrough(authenticated(call, upstream.uri()));

    route
        .execute(host.request().unwrap(), &host, None)
        .await
        .unwrap();

    assert!(answer.0.lock().unwrap().is_empty());
    assert_eq!(received(&upstream).await.len(), 1);
}

#[derive(Default)]
struct DecliningAnswer(Mutex<Vec<bool>>);

impl litellm_inference_messages::HostAnswer for DecliningAnswer {
    fn answer(&self, _stream: bool) -> futures_util::future::BoxFuture<'_, Result<(), Error>> {
        panic!("a declined interception never asks for an answer")
    }

    fn intercept(&self, stream: bool) -> futures_util::future::BoxFuture<'_, Result<bool, Error>> {
        self.0.lock().unwrap().push(stream);
        Box::pin(async { Ok(false) })
    }
}

#[rstest]
#[tokio::test]
async fn an_interception_no_interceptor_takes_reaches_the_provider(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let answer = Arc::new(DecliningAnswer::default());
    let route = messages_route(no_secrets()).with_host_answer(answer.clone());
    let mut request = authenticated(call, upstream.uri());
    request.shaping.settings.intercepted = true;
    let host = RecordingHost::passthrough(request);

    route
        .execute(host.request().unwrap(), &host, None)
        .await
        .unwrap();

    assert_eq!(*answer.0.lock().unwrap(), vec![false]);
    assert_eq!(received(&upstream).await.len(), 1);
}
