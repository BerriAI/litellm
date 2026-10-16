use std::{convert::Infallible, time::Duration};

use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    interceptors::{
        Cost, ExecutionFacts, Interceptors, ProviderIdentity, RequestContext, ResultSource,
        WireRequest,
    },
    observation::ObservationSender,
    protocol::Protocol,
};
use litellm_http::{outbound::OutboundRequest, request::truncate_error_body, transport};
use litellm_inference::{CallOptions, context::CallContext, provider::resolve_llm_provider};
use litellm_llms::{
    base_llm::auth::resolve_auth,
    openai::chat::transformation::{OPENAI_CHAT_COMPLETIONS_CONFIG, is_stream},
};
use serde_json::{Map, Value};

use crate::{
    ChatCompletionsRoute, Error, common_utils::string_headers,
    constants::CHAT_COMPLETIONS_TIMEOUT_SECS,
};

pub struct OpenAiChatCompletions;

impl Protocol for OpenAiChatCompletions {
    type Request = OpenAiChatCall;
    /// The provider's JSON body, as it sent it.
    type Response = Bytes;
    type Error = Error;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

pub type OpenAiChatOutput = CallOutput<Bytes, (), Bytes, Error>;

/// A Chat Completions body bound for an OpenAI deployment. `body` is the caller's request;
/// `model` is the deployment's, with or without its `openai/` prefix.
pub struct OpenAiChatCall {
    pub model: String,
    pub body: Map<String, Value>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
}

impl ChatCompletionsRoute {
    pub fn serves_openai(model: &str, custom_llm_provider: Option<&str>) -> bool {
        resolve_llm_provider(model, custom_llm_provider, "chat completions")
            .is_ok_and(|resolved| resolved.provider == LlmProviders::Openai)
    }

    pub fn openai_machine(
        self,
        call: OpenAiChatCall,
        observers: Option<ObservationSender>,
    ) -> HostedMachine<OpenAiChatCompletions> {
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                let context = CallContext::new(
                    &interceptors,
                    CallOptions {
                        cache: None,
                        observers,
                    },
                );
                Box::pin(self.run_openai(call, context)).await
            },
        )
    }

    pub async fn execute_openai(
        &self,
        call: OpenAiChatCall,
        interceptors: &impl Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<OpenAiChatOutput, Error> {
        let context = CallContext::new(
            interceptors,
            CallOptions {
                cache: None,
                observers,
            },
        );
        litellm_host::lifecycle::observe_call(
            context.observers.clone(),
            Box::pin(self.run_openai(call, context)),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %call.model,
        provider,
        resolved_model,
        stream,
        outcome
    ))]
    async fn run_openai(
        &self,
        call: OpenAiChatCall,
        context: CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<OpenAiChatOutput, Error> {
        litellm_inference::diagnostic::call(async {
            let resolved = resolve_llm_provider(
                &call.model,
                call.custom_llm_provider.as_deref(),
                "chat completions",
            )?;
            if resolved.provider != LlmProviders::Openai {
                return Err(Error::InvalidProvider(
                    <&str>::from(resolved.provider).to_string(),
                ));
            }
            let model = resolved.model.to_string();
            let provider = <&str>::from(LlmProviders::Openai);
            litellm_inference::diagnostic::provider(&model, provider);
            let config = &OPENAI_CHAT_COMPLETIONS_CONFIG;
            let secrets = self
                .secrets
                .resolve(config.secret_names(call.api_key.as_deref(), call.api_base.as_deref()))
                .await?;
            let lookup = |key: &str| secrets.get(key);
            let body = config.transform_request(&model, call.body)?;
            let environment = config.validate_environment(
                string_headers(call.extra_headers)?,
                call.api_key.as_deref(),
                &lookup,
            )?;
            let url = config.get_complete_url(call.api_base.as_deref(), &lookup);
            let authenticated = resolve_auth(&self.auth, environment, &lookup).await?;
            let request_context = RequestContext {
                model: model.clone(),
                custom_llm_provider: provider.to_string(),
                optional_params: Value::Object(
                    body.iter()
                        .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .collect(),
                ),
                secret_fields: Vec::new(),
                api_key: call.api_key.map(litellm_auth::SecretValue::new),
            };
            let wire = context
                .interceptors
                .before_provider_request(
                    WireRequest {
                        url,
                        headers: authenticated.headers,
                        body: Value::Object(body),
                    },
                    request_context,
                )
                .await?;
            let stream = wire
                .body
                .as_object()
                .map(is_stream)
                .transpose()?
                .unwrap_or(false);
            let timeout = call
                .timeout
                .unwrap_or(Duration::from_secs(CHAT_COMPLETIONS_TIMEOUT_SECS));
            let response = send(&self.http, wire, stream, timeout).await?;
            let output = if stream {
                CallOutput::Stream {
                    head: (),
                    chunks: chunks(response, timeout),
                }
            } else {
                let body = response.bytes().await.map_err(network)?;
                context
                    .response_received(&String::from_utf8_lossy(&body))
                    .await?;
                CallOutput::Complete(body)
            };
            context
                .result_ready(ExecutionFacts {
                    provider: ProviderIdentity {
                        model,
                        provider: provider.to_string(),
                    },
                    source: ResultSource::Provider,
                    cost: Cost::Deferred,
                })
                .await?;
            Ok(output)
        })
        .await
    }
}

/// A non-streaming call keeps the usual total timeout. A stream has no total limit, so a
/// 30 minute generation is not cut off: `timeout` bounds the wait for the response head
/// here and each read after it in [`chunks`].
async fn send(
    http: &litellm_http::Client,
    wire: WireRequest,
    stream: bool,
    timeout: Duration,
) -> Result<reqwest::Response, Error> {
    let request = OutboundRequest::json(
        wire.url,
        wire.headers,
        &wire.body,
        (!stream).then_some(timeout),
    )?;
    let response = tokio::time::timeout(timeout, litellm_inference::outbound::send(request, http))
        .await
        .map_err(|_| silent_provider(timeout))?
        .map_err(|error| Error::Transport(transport::Error::from_reqwest_before_dispatch(error)))?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let text = response.text().await.map_err(network)?;
    Err(Error::Transport(transport::Error::Http {
        status: status.as_u16(),
        body: truncate_error_body(&text),
    }))
}

fn chunks(response: reqwest::Response, idle: Duration) -> BoxStream<'static, Result<Bytes, Error>> {
    futures_util::stream::try_unfold(response, move |mut response| async move {
        let chunk = tokio::time::timeout(idle, response.chunk())
            .await
            .map_err(|_| silent_provider(idle))?
            .map_err(network)?;
        Ok(chunk.map(|chunk| (chunk, response)))
    })
    .boxed()
}

fn network(error: reqwest::Error) -> Error {
    Error::Transport(transport::Error::from(error))
}

fn silent_provider(timeout: Duration) -> Error {
    Error::Transport(transport::Error::Network(format!(
        "the provider sent nothing for {}s",
        timeout.as_secs_f64()
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::TryStreamExt;
    use litellm_host::interceptors::RawResponse;
    use rstest::{fixture, rstest};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{header, method, path},
    };

    use super::*;

    const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-6-astra","system_fingerprint":"fp","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#;
    const STREAM: &str = "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n";

    #[derive(Default)]
    struct Recording {
        contexts: Mutex<Vec<RequestContext>>,
        raw: Mutex<Vec<String>>,
        results: Mutex<Vec<ExecutionFacts>>,
    }

    impl Interceptors<Error> for Recording {
        async fn before_provider_request(
            &self,
            wire: WireRequest,
            context: RequestContext,
        ) -> Result<WireRequest, Error> {
            self.contexts.lock().unwrap().push(context);
            Ok(wire)
        }

        async fn after_provider_response(&self, raw: RawResponse) -> Result<(), Error> {
            self.raw.lock().unwrap().push(raw.body);
            Ok(())
        }

        async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
            self.results.lock().unwrap().push(facts);
            Ok(())
        }
    }

    #[fixture]
    fn route() -> ChatCompletionsRoute {
        ChatCompletionsRoute::new(
            litellm_http::Client::plain_for_test(),
            Arc::new(litellm_auth::AuthServices::default()),
            litellm_inference_testing::no_secrets(),
        )
    }

    fn call(api_base: &str, body: Value) -> OpenAiChatCall {
        let Value::Object(body) = body else {
            panic!("the body is an object");
        };
        OpenAiChatCall {
            model: "openai/gpt-6-astra".into(),
            body,
            api_key: Some("sk-deployment".into()),
            api_base: Some(format!("{api_base}/v1")),
            custom_llm_provider: None,
            extra_headers: None,
            timeout: Some(Duration::from_secs(5)),
        }
    }

    async fn run(
        route: &ChatCompletionsRoute,
        call: OpenAiChatCall,
        interceptors: &Recording,
    ) -> Result<OpenAiChatOutput, Error> {
        route
            .run_openai(call, CallContext::new(interceptors, CallOptions::default()))
            .await
    }

    async fn drain(chunks: BoxStream<'static, Result<Bytes, Error>>) -> Result<Vec<u8>, Error> {
        Ok(chunks.try_collect::<Vec<_>>().await?.concat())
    }

    async fn collect(output: OpenAiChatOutput) -> Result<Vec<u8>, Error> {
        match output {
            CallOutput::Complete(body) => panic!("expected a stream, got {body:?}"),
            CallOutput::Stream { chunks, .. } => drain(chunks).await,
        }
    }

    #[rstest]
    #[case::openai_prefix("openai/gpt-6-astra", None, true)]
    #[case::explicit_provider("gpt-6-astra", Some("openai"), true)]
    #[case::openai_like("openai_like/gpt-6-astra", None, false)]
    #[case::anthropic("anthropic/claude", None, false)]
    #[case::unprefixed("gpt-6-astra", None, false)]
    fn only_openai_deployments_take_the_passthrough(
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            ChatCompletionsRoute::serves_openai(model, provider),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_completion_returns_the_provider_body_byte_for_byte(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer sk-deployment"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let interceptors = Recording::default();

        let output = run(
            &route,
            call(
                &upstream.uri(),
                json!({"model": "public", "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "f"}}], "metadata": {"k": "v"}}),
            ),
            &interceptors,
        )
        .await
        .unwrap();

        let CallOutput::Complete(body) = output else {
            panic!("a non-streaming call completes");
        };
        assert_eq!(body, COMPLETION.as_bytes());
        let requests = upstream.received_requests().await.unwrap();
        let sent: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(
            sent,
            json!({"model": "gpt-6-astra", "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "f"}}]})
        );
        assert_eq!(*interceptors.raw.lock().unwrap(), [COMPLETION]);
        let contexts = interceptors.contexts.lock().unwrap();
        assert_eq!(
            (
                contexts[0].model.as_str(),
                contexts[0].custom_llm_provider.as_str()
            ),
            ("gpt-6-astra", "openai")
        );
        assert_eq!(
            contexts[0].optional_params,
            json!({"tools": [{"type": "function", "function": {"name": "f"}}]})
        );
        assert_eq!(interceptors.results.lock().unwrap().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_relays_every_byte_including_the_usage_chunk(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(STREAM, "text/event-stream"))
            .mount(&upstream)
            .await;
        let interceptors = Recording::default();

        let output = run(
            &route,
            call(
                &upstream.uri(),
                json!({"model": "public", "messages": [{"role": "user", "content": "hi"}],
                    "stream": true, "stream_options": {"include_usage": true}}),
            ),
            &interceptors,
        )
        .await
        .unwrap();

        assert_eq!(collect(output).await.unwrap(), STREAM.as_bytes());
        let request: &Request = &upstream.received_requests().await.unwrap()[0];
        let sent: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(sent["stream_options"], json!({"include_usage": true}));
        assert!(interceptors.raw.lock().unwrap().is_empty());
    }

    /// Serves an SSE head and the first event, then waits for `release` before the rest.
    async fn paused_upstream(release: tokio::sync::oneshot::Receiver<()>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 64 * 1024];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            let event = "data: {\"n\":1}\n\n";
            socket
                .write_all(format!("{:x}\r\n{event}\r\n", event.len()).as_bytes())
                .await
                .unwrap();
            let _ = release.await;
            let done = "data: [DONE]\n\n";
            let _ = socket
                .write_all(format!("{:x}\r\n{done}\r\n0\r\n\r\n", done.len()).as_bytes())
                .await;
        });
        base
    }

    #[rstest]
    #[tokio::test]
    async fn each_event_is_delivered_before_the_stream_finishes(route: ChatCompletionsRoute) {
        let (release, released) = tokio::sync::oneshot::channel();
        let base = paused_upstream(released).await;
        let output = run(
            &route,
            call(&base, json!({"model": "m", "messages": [], "stream": true})),
            &Recording::default(),
        )
        .await
        .unwrap();
        let CallOutput::Stream { mut chunks, .. } = output else {
            panic!("a streaming call streams");
        };

        let first = tokio::time::timeout(Duration::from_secs(2), chunks.try_next())
            .await
            .expect("the first event arrives while the provider is still streaming")
            .unwrap()
            .unwrap();
        assert_eq!(first, "data: {\"n\":1}\n\n");
        release.send(()).unwrap();
        assert_eq!(drain(chunks).await.unwrap(), b"data: [DONE]\n\n");
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_outlives_the_request_timeout_while_data_keeps_coming(
        route: ChatCompletionsRoute,
    ) {
        let (release, released) = tokio::sync::oneshot::channel();
        let base = paused_upstream(released).await;
        let mut streaming = call(&base, json!({"model": "m", "messages": [], "stream": true}));
        streaming.timeout = Some(Duration::from_millis(300));
        let output = run(&route, streaming, &Recording::default()).await.unwrap();
        let CallOutput::Stream { mut chunks, .. } = output else {
            panic!("a streaming call streams");
        };
        chunks.try_next().await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        release.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;

        assert_eq!(drain(chunks).await.unwrap(), b"data: [DONE]\n\n");
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_that_goes_silent_past_the_timeout_fails(route: ChatCompletionsRoute) {
        let (_release, released) = tokio::sync::oneshot::channel();
        let base = paused_upstream(released).await;
        let mut streaming = call(&base, json!({"model": "m", "messages": [], "stream": true}));
        streaming.timeout = Some(Duration::from_millis(100));
        let output = run(&route, streaming, &Recording::default()).await.unwrap();

        assert!(matches!(
            collect(output).await,
            Err(Error::Transport(transport::Error::Network(_)))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn an_upstream_error_keeps_its_status(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        let body = r#"{"error":{"message":"slow down","type":"rate_limit_error"}}"#;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(body, "application/json"))
            .mount(&upstream)
            .await;

        let error = run(
            &route,
            call(
                &upstream.uri(),
                json!({"model": "m", "messages": [], "stream": true}),
            ),
            &Recording::default(),
        )
        .await
        .err()
        .expect("the upstream failure fails the call");

        assert_eq!(
            error,
            Error::Transport(transport::Error::Http {
                status: 429,
                body: body.into()
            })
        );
    }

    #[rstest]
    #[tokio::test]
    async fn the_key_falls_back_to_the_environment(route: ChatCompletionsRoute) {
        let upstream = MockServer::start().await;
        Mock::given(header("authorization", "Bearer sk-env"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json"))
            .expect(1)
            .mount(&upstream)
            .await;
        let route = ChatCompletionsRoute {
            secrets: Arc::new(litellm_inference_testing::RecordingSecrets::new([(
                "OPENAI_API_KEY",
                "sk-env",
            )])),
            ..route
        };
        let mut without_key = call(&upstream.uri(), json!({"model": "m", "messages": []}));
        without_key.api_key = None;

        run(&route, without_key, &Recording::default())
            .await
            .unwrap();
    }

    #[rstest]
    #[tokio::test]
    async fn a_missing_key_fails_before_any_request(route: ChatCompletionsRoute) {
        let mut without_key = call("http://127.0.0.1:1", json!({"model": "m", "messages": []}));
        without_key.api_key = None;
        assert!(matches!(
            run(&route, without_key, &Recording::default()).await,
            Err(Error::Auth(litellm_auth::Error::MissingApiKey { .. }))
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn a_non_openai_deployment_is_refused(route: ChatCompletionsRoute) {
        let mut other = call("http://127.0.0.1:1", json!({"model": "m", "messages": []}));
        other.model = "openai_like/gpt".into();
        assert_eq!(
            run(&route, other, &Recording::default()).await.err(),
            Some(Error::InvalidProvider("openai_like".into()))
        );
    }
}
