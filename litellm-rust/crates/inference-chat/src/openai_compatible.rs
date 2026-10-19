//! The OpenAI-compatible providers (`litellm/llms/<provider>/chat/transformation.py`) through
//! the OpenAI passthrough: each provider's config shapes the request, and the response goes
//! back as the provider sent it but for the fixups its config makes. A stream is relayed
//! event by event, with the usage the call asked for on the client's behalf recorded for the
//! host and kept from the client. When the client asked for usage and the provider sent none,
//! the usage is estimated from the request and the streamed text, as Python's stream wrapper
//! does.

use std::{collections::BTreeMap, sync::LazyLock};

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, stream::BoxStream};
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_framer::sse::{SseBlock, SseBlockCodec};
use litellm_llms::{
    ErrorDetail,
    base_llm::{
        auth::{AuthScheme, ValidatedEnvironment, resolve_auth},
        sse::DONE_FRAME,
    },
    cerebras::chat::transformation::CEREBRAS_CHAT_CONFIG,
    deepseek::chat::transformation::DEEPSEEK_CHAT_CONFIG,
    fireworks_ai::chat::transformation::FIREWORKS_AI_CHAT_CONFIG,
    groq::chat::transformation::GROQ_CHAT_CONFIG,
    hosted_vllm::chat::transformation::HOSTED_VLLM_CHAT_CONFIG,
    mistral::chat::transformation::MISTRAL_CHAT_CONFIG,
    nvidia_nim::chat::transformation::NVIDIA_NIM_CHAT_CONFIG,
    openai::chat::compatible::{
        CompatibleChatBody, CompatibleContext, OpenAiCompatibleChatConfig, ResponseShape,
        fix_response, transform_request,
    },
    openrouter::chat::transformation::OPENROUTER_CHAT_CONFIG,
    perplexity::chat::transformation::PERPLEXITY_CHAT_CONFIG,
    together_ai::chat::transformation::TOGETHER_AI_CHAT_CONFIG,
    xai::chat::transformation::XAI_CHAT_CONFIG,
};
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionsUsage, CompletionTokensDetails, PromptTokensDetails,
};
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::{Map, Value, json};
use tokio_util::codec::Decoder;

use crate::{
    ChatCompletionsRoute, Error,
    common_utils::string_headers,
    openai::{OpenAiChatCall, PreparedCall},
    stream::ReportedUsage,
};

const ENV_REFERENCE: &str = "os.environ/";

/// The tokenizer Python's `token_counter` falls back to for models it has no tokenizer for.
static TOKENIZER: LazyLock<Option<TokenCounter>> =
    LazyLock::new(|| TokenCounter::from_tiktoken("cl100k_base").ok());

/// The config of a provider served through the OpenAI passthrough with its own transforms.
pub fn compatible_chat_config(
    provider: LlmProviders,
) -> Option<&'static dyn OpenAiCompatibleChatConfig> {
    Some(match provider {
        LlmProviders::Cerebras => &CEREBRAS_CHAT_CONFIG,
        LlmProviders::Deepseek => &DEEPSEEK_CHAT_CONFIG,
        LlmProviders::FireworksAi => &FIREWORKS_AI_CHAT_CONFIG,
        LlmProviders::Groq => &GROQ_CHAT_CONFIG,
        LlmProviders::HostedVllm => &HOSTED_VLLM_CHAT_CONFIG,
        LlmProviders::Mistral => &MISTRAL_CHAT_CONFIG,
        LlmProviders::NvidiaNim => &NVIDIA_NIM_CHAT_CONFIG,
        LlmProviders::Openrouter => &OPENROUTER_CHAT_CONFIG,
        LlmProviders::Perplexity => &PERPLEXITY_CHAT_CONFIG,
        LlmProviders::TogetherAi => &TOGETHER_AI_CHAT_CONFIG,
        LlmProviders::Xai => &XAI_CHAT_CONFIG,
        _ => return None,
    })
}

/// What the response of a call to an OpenAI-compatible provider goes through on its way back.
#[derive(Clone)]
pub(crate) struct ResponseAdapter {
    config: &'static dyn OpenAiCompatibleChatConfig,
    shape: ResponseShape,
    /// The request's prompt tokens, counted when the client asked for stream usage, for a
    /// provider that streams none.
    prompt_tokens: Option<u64>,
}

/// What a stream has said so far, for the usage chunk estimated when the provider sends none.
#[derive(Default)]
struct Streamed {
    text: String,
    reasoning: String,
    /// Tool calls by choice and call index: id, name and the arguments streamed so far.
    tool_calls: BTreeMap<(u64, u64), [String; 3]>,
    usage_seen: bool,
    last: Map<String, Value>,
}

impl ChatCompletionsRoute {
    pub(crate) async fn prepare_openai_compatible(
        &self,
        config: &'static dyn OpenAiCompatibleChatConfig,
        model: String,
        call: OpenAiChatCall,
    ) -> Result<PreparedCall, Error> {
        let OpenAiChatCall {
            body,
            api_key,
            api_base,
            extra_headers,
            drop_params,
            model_info,
            ..
        } = call;
        let references: Vec<&str> = [api_key.as_deref(), api_base.as_deref()]
            .into_iter()
            .flatten()
            .filter_map(|value| value.strip_prefix(ENV_REFERENCE))
            .collect();
        let names: Vec<&str> = config
            .secret_names()
            .into_iter()
            .chain(references)
            .collect();
        let secrets = self.secrets.resolve(&names).await?;
        let lookup = |name: &str| secrets.get(name);
        // Python's router reads every `os.environ/` setting from the environment when it
        // loads the deployment. A reference to an unset variable counts as unset.
        let resolve = |value: &str| match value.strip_prefix(ENV_REFERENCE) {
            Some(name) => lookup(name),
            None => Some(value.to_owned()),
        };
        let api_key = api_key.as_deref().and_then(resolve);
        let api_base = api_base.as_deref().and_then(resolve);
        let url = config.complete_url(&config.api_base(api_base.as_deref(), &lookup)?);
        let key = config.api_key(api_key.as_deref(), &lookup)?;
        let provider_headers = config.headers(&body, &lookup);
        let context = CompatibleContext {
            model: &model,
            drop_params,
            model_info: model_info.as_deref(),
        };
        let prompt_tokens = wants_stream_usage(&body).then(|| prompt_tokens(&body));
        let CompatibleChatBody { body, shape } = transform_request(config, &context, body)?;
        let defaults: Vec<(&str, &str)> = provider_headers
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .chain([("content-type", "application/json")])
            .collect();
        let environment = ValidatedEnvironment {
            headers: litellm_http::request::with_default_headers(
                string_headers(extra_headers)?,
                &defaults,
            ),
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(key),
            },
        };
        let authenticated = resolve_auth(&self.auth, environment, &lookup).await?;
        Ok(PreparedCall {
            model,
            provider: config.name(),
            url,
            headers: authenticated.headers,
            body,
            api_key,
            adapter: Some(ResponseAdapter {
                config,
                shape,
                prompt_tokens,
            }),
        })
    }
}

impl ResponseAdapter {
    /// A complete response, re-encoded only when a fixup changed it.
    pub(crate) fn complete(&self, body: Bytes) -> Bytes {
        let Ok(Value::Object(mut response)) = serde_json::from_slice::<Value>(&body) else {
            return body;
        };
        match fix_response(self.config, &mut response, &self.shape) {
            true => serde_json::to_vec(&response).map_or(body, Bytes::from),
            false => body,
        }
    }

    pub(crate) fn stream(
        self,
        upstream: BoxStream<'static, Result<Bytes, Error>>,
        usage: ReportedUsage,
    ) -> BoxStream<'static, Result<Bytes, Error>> {
        let relay = Relay {
            upstream,
            buffer: BytesMut::new(),
            codec: SseBlockCodec::default(),
            adapter: self,
            usage,
            streamed: Streamed::default(),
            ended: false,
        };
        futures_util::stream::try_unfold(relay, |mut relay| async move {
            relay
                .next_frame()
                .await
                .map(|frame| frame.map(|frame| (frame, relay)))
        })
        .boxed()
    }

    /// One event's frame for the client, or `None` when the client is not to see it.
    fn frame(
        &self,
        block: SseBlock,
        usage: &ReportedUsage,
        streamed: &mut Streamed,
    ) -> Option<Bytes> {
        if block.event.data.trim() == "[DONE]" {
            return Some(match self.estimated_usage(usage, streamed) {
                Some(frame) => [frame.as_slice(), DONE_FRAME].concat().into(),
                None => Bytes::from_static(DONE_FRAME),
            });
        }
        let Ok(Value::Object(mut chunk)) = serde_json::from_str::<Value>(&block.event.data) else {
            return Some(block.bytes);
        };
        let mut changed = self.config.fix_chunk(&mut chunk, &self.shape);
        if let Some(reported) = chunk.get("usage").and_then(usage_from_json) {
            usage.set(reported);
            streamed.usage_seen = true;
        }
        if self.prompt_tokens.is_some() {
            streamed.record(&chunk);
        }
        if !self.shape.client_usage {
            if chunk.get("usage").is_some_and(|usage| !usage.is_null()) {
                chunk.remove("usage");
                changed = true;
            }
            if chunk
                .get("choices")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                return None;
            }
        }
        if !changed {
            return Some(block.bytes);
        }
        sse_frame(&chunk).map(Bytes::from)
    }

    /// The usage chunk Python's stream wrapper adds when the client asked for usage and the
    /// provider sent none: the request's prompt tokens, and the tokens of the streamed content
    /// and tool calls plus those of the reasoning, counted apart as Python counts them, on a
    /// chunk shaped like the stream's own. It is recorded as the call's usage too.
    fn estimated_usage(&self, usage: &ReportedUsage, streamed: &Streamed) -> Option<Vec<u8>> {
        let prompt_tokens = self.prompt_tokens?;
        if streamed.usage_seen {
            return None;
        }
        let tokenizer = TOKENIZER.as_ref()?;
        let reasoning_tokens = match streamed.reasoning.is_empty() {
            true => 0,
            false => tokenizer.count_text(&streamed.reasoning).ok()?,
        };
        let completion_tokens =
            (tokenizer.count_text(&streamed.completion()).ok()? + reasoning_tokens) as u64;
        let counts = json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        });
        usage.set(usage_from_json(&counts)?);
        let mut chunk = Map::new();
        for field in ["id", "object", "created", "model"] {
            if let Some(value) = streamed.last.get(field) {
                chunk.insert(field.into(), value.clone());
            }
        }
        chunk.insert("choices".into(), json!([{"index": 0, "delta": {}}]));
        chunk.insert("usage".into(), counts);
        sse_frame(&chunk)
    }
}

/// The string at `value`, or empty.
fn text(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or_default()
}

impl Streamed {
    /// Keeps the chunk's identity fields and what it streamed: content, reasoning, and each
    /// tool call's id, name and arguments.
    fn record(&mut self, chunk: &Map<String, Value>) {
        for field in ["id", "object", "created", "model"] {
            if let Some(value) = chunk.get(field) {
                self.last.insert(field.into(), value.clone());
            }
        }
        for (position, choice) in chunk
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            let index = choice
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64);
            self.text.push_str(text(delta.get("content")));
            self.reasoning
                .push_str(text(delta.get("reasoning_content")));
            let calls = delta.get("tool_calls").and_then(Value::as_array);
            for (order, call) in calls.into_iter().flatten().enumerate() {
                let slot = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(order as u64);
                let [id, name, arguments] = self.tool_calls.entry((index, slot)).or_default();
                id.push_str(text(call.get("id")));
                name.push_str(text(call.pointer("/function/name")));
                arguments.push_str(text(call.pointer("/function/arguments")));
            }
        }
    }

    /// The text Python counts as the completion: the content, then each tool call as its
    /// `model_dump_json()` writes it.
    fn completion(&self) -> String {
        let quoted = |text: &str| Value::from(text).to_string();
        self.tool_calls.values().fold(
            self.text.clone(),
            |mut completion, [id, name, arguments]| {
                completion.push_str(&format!(
                    r#"{{"function":{{"arguments":{},"name":{}}},"id":{},"type":"function"}}"#,
                    quoted(arguments),
                    quoted(name),
                    quoted(id),
                ));
                completion
            },
        )
    }
}

fn sse_frame(chunk: &Map<String, Value>) -> Option<Vec<u8>> {
    let mut frame = b"data: ".to_vec();
    serde_json::to_writer(&mut frame, chunk).ok()?;
    frame.extend_from_slice(b"\n\n");
    Some(frame)
}

/// Whether the request streams and asks to see the usage.
fn wants_stream_usage(body: &Map<String, Value>) -> bool {
    body.get("stream") == Some(&Value::Bool(true))
        && body
            .get("stream_options")
            .and_then(|options| options.get("include_usage"))
            == Some(&Value::Bool(true))
}

/// The request's prompt tokens as Python's stream wrapper counts them, from the messages alone
/// (`token_counter(model, messages)`), or 0 when they can't be counted.
fn prompt_tokens(body: &Map<String, Value>) -> u64 {
    let request: Map<String, Value> = body
        .get("messages")
        .map(|messages| ("messages".to_owned(), messages.clone()))
        .into_iter()
        .collect();
    let counted = serde_json::from_value::<CountableRequest>(Value::Object(request))
        .ok()
        .and_then(|request| TOKENIZER.as_ref()?.count_request(&request).ok());
    counted.map_or(0, |count| count.input_tokens as u64)
}

struct Relay {
    upstream: BoxStream<'static, Result<Bytes, Error>>,
    buffer: BytesMut,
    codec: SseBlockCodec,
    adapter: ResponseAdapter,
    usage: ReportedUsage,
    streamed: Streamed,
    ended: bool,
}

impl Relay {
    async fn next_frame(&mut self) -> Result<Option<Bytes>, Error> {
        loop {
            let block = match self.ended {
                false => self.codec.decode(&mut self.buffer),
                true => self.codec.decode_eof(&mut self.buffer),
            }
            .map_err(|error| {
                Error::InvalidResponse(ErrorDetail::from(format!(
                    "the provider sent a malformed stream: {error}"
                )))
            })?;
            match block {
                Some(block) => {
                    if let Some(frame) = self.adapter.frame(block, &self.usage, &mut self.streamed)
                    {
                        return Ok(Some(frame));
                    }
                }
                None if self.ended => return Ok(None),
                None => match self.upstream.next().await {
                    Some(bytes) => self.buffer.extend_from_slice(&bytes?),
                    None => self.ended = true,
                },
            }
        }
    }
}

/// The usage a chunk or response reports, read as leniently as Python's `Usage` reads it.
fn usage_from_json(usage: &Value) -> Option<ChatCompletionsUsage> {
    let usage = usage.as_object()?;
    let count = |value: Option<&Value>| value.and_then(Value::as_u64);
    let prompt = usage.get("prompt_tokens_details");
    let completion = usage.get("completion_tokens_details");
    Some(ChatCompletionsUsage {
        prompt_tokens: count(usage.get("prompt_tokens")).unwrap_or(0),
        completion_tokens: count(usage.get("completion_tokens")).unwrap_or(0),
        total_tokens: count(usage.get("total_tokens")).unwrap_or(0),
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: count(prompt.and_then(|d| d.get("cached_tokens"))).unwrap_or(0),
            cache_creation_tokens: count(prompt.and_then(|d| d.get("cache_creation_tokens")))
                .unwrap_or(0),
            text_tokens: count(prompt.and_then(|d| d.get("text_tokens"))).unwrap_or(0),
            cache_write_tokens: None,
            cache_creation_token_details: prompt
                .and_then(|d| d.get("cache_creation_token_details"))
                .filter(|details| details.is_object())
                .cloned(),
        },
        completion_tokens_details: completion.filter(|details| details.is_object()).map(
            |details| CompletionTokensDetails {
                reasoning_tokens: count(details.get("reasoning_tokens")),
                text_tokens: count(details.get("text_tokens")),
            },
        ),
        server_tool_use: usage
            .get("server_tool_use")
            .filter(|tools| tools.is_object())
            .cloned(),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn adapter(
        config: &'static dyn OpenAiCompatibleChatConfig,
        client_usage: bool,
    ) -> ResponseAdapter {
        ResponseAdapter {
            config,
            shape: ResponseShape {
                stream: true,
                client_usage,
                ..ResponseShape::default()
            },
            prompt_tokens: None,
        }
    }

    async fn relay(adapter: ResponseAdapter, parts: &[&'static str]) -> (String, ReportedUsage) {
        let usage = ReportedUsage::default();
        let upstream = futures_util::stream::iter(
            parts
                .iter()
                .map(|part| Ok::<_, Error>(Bytes::from_static(part.as_bytes())))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let frames: Vec<Bytes> = adapter
            .stream(upstream, usage.clone())
            .map(Result::unwrap)
            .collect()
            .await;
        (String::from_utf8(frames.concat()).unwrap(), usage)
    }

    const CONTENT: &str =
        "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n";
    const USAGE: &str = "data: {\"id\":\"c\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n";
    const DONE: &str = "data: [DONE]\n\n";

    #[rstest]
    #[case::kept_from_the_client(false, format!("{CONTENT}{DONE}"))]
    #[case::shown_when_asked(true, format!("{CONTENT}{USAGE}{DONE}"))]
    #[tokio::test]
    async fn usage_is_recorded_and_shown_only_when_asked(
        #[case] client_usage: bool,
        #[case] expected: String,
    ) {
        let (sent, usage) = relay(
            adapter(&GROQ_CHAT_CONFIG, client_usage),
            &[
                "data: {\"id\":\"c\",\"choices\":[{\"index\":0,",
                "\"delta\":{\"content\":\"hi\"}}]}\n\n",
                USAGE,
                DONE,
            ],
        )
        .await;
        assert_eq!(sent, expected);
        let usage = usage.get().unwrap();
        assert_eq!(
            (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens
            ),
            (3, 2, 5)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn chunk_fixups_reencode_only_the_chunks_they_change() {
        let (sent, _) = relay(
            adapter(&HOSTED_VLLM_CHAT_CONFIG, true),
            &[
                CONTENT,
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"r\"}}]}\n\n",
                DONE,
            ],
        )
        .await;
        assert_eq!(
            sent,
            format!(
                "{CONTENT}data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":\"r\"}}}}]}}\n\n{DONE}"
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_usage_chunk_with_choices_loses_only_its_usage() {
        let (sent, usage) = relay(
            adapter(&MISTRAL_CHAT_CONFIG, false),
            &["data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"],
        )
        .await;
        assert_eq!(
            sent,
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        );
        assert_eq!(usage.get().unwrap().total_tokens, 2);
    }

    #[rstest]
    #[tokio::test]
    async fn usage_the_provider_did_not_send_is_estimated_when_the_client_asked() {
        let body = json!({"messages": [{"role": "user", "content": "Say hello."}], "stream": true,
            "stream_options": {"include_usage": true}});
        let body = body.as_object().unwrap();
        assert!(wants_stream_usage(body));
        let adapter = ResponseAdapter {
            prompt_tokens: Some(prompt_tokens(body)),
            ..adapter(&MISTRAL_CHAT_CONFIG, true)
        };
        let (sent, usage) = relay(
            adapter,
            &[
                "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello from the parity mock.\"}}]}\n\n",
                DONE,
            ],
        )
        .await;
        // Python's counts for the same request and text: 3 per message, 1 for the role, the
        // text's 3, and 3 priming the reply; the answer is 6 tokens.
        assert!(
            sent.ends_with("data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":6,\"total_tokens\":16}}\n\ndata: [DONE]\n\n"),
            "{sent}"
        );
        assert_eq!(usage.get().unwrap().total_tokens, 16);
    }

    #[rstest]
    #[tokio::test]
    async fn an_estimate_counts_tool_calls_as_python_dumps_them_and_not_the_tools() {
        let body = json!({"messages": [{"role": "user", "content": "Weather in Paris? [[parity reply=tool]]"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "stream": true, "stream_options": {"include_usage": true}});
        let adapter = ResponseAdapter {
            prompt_tokens: Some(prompt_tokens(body.as_object().unwrap())),
            ..adapter(&MISTRAL_CHAT_CONFIG, true)
        };
        let (_, usage) = relay(
            adapter,
            &[
                "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_parity_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\"\"}}]}}]}\n\n",
                "data: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"Paris\\\"}\"}}]}}]}\n\n",
                DONE,
            ],
        )
        .await;
        // Python's counts for the same stream: the messages without the tools, and the call as
        // `{"function":{"arguments":"{\"city\": \"Paris\"}","name":"get_weather"},"id":"call_parity_1","type":"function"}`.
        let usage = usage.get().unwrap();
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (17, 30));
    }

    #[rstest]
    #[tokio::test]
    async fn usage_the_provider_sent_is_not_estimated() {
        let adapter = ResponseAdapter {
            prompt_tokens: Some(10),
            ..adapter(&MISTRAL_CHAT_CONFIG, true)
        };
        let (sent, usage) = relay(adapter, &[CONTENT, USAGE, DONE]).await;
        assert_eq!(sent, format!("{CONTENT}{USAGE}{DONE}"));
        assert_eq!(usage.get().unwrap().total_tokens, 5);
    }

    #[rstest]
    fn a_complete_response_is_reencoded_only_when_fixed() {
        let raw = Bytes::from_static(
            br#"{"choices": [{"message": {"role": "assistant", "content": "x"}}]}"#,
        );
        assert_eq!(
            adapter(&CEREBRAS_CHAT_CONFIG, false).complete(raw.clone()),
            raw
        );
        let fixed = adapter(&HOSTED_VLLM_CHAT_CONFIG, false).complete(Bytes::from_static(
            br#"{"choices":[{"message":{"role":"assistant","content":"x","reasoning":"r"}}]}"#,
        ));
        assert_eq!(
            serde_json::from_slice::<Value>(&fixed).unwrap(),
            json!({"choices": [{"message": {"role": "assistant", "content": "x", "reasoning_content": "r"}}]})
        );
    }
}
