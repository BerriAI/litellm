//! The OpenAI-compatible providers (`litellm/llms/<provider>/chat/transformation.py`) through
//! the OpenAI passthrough: each provider's config shapes the request, and the response goes
//! back as the provider sent it but for the fixups its config makes. A stream is relayed
//! event by event, with the usage the call asked for on the client's behalf recorded for the
//! host and kept from the client.

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
use serde_json::Value;
use tokio_util::codec::Decoder;

use crate::{
    ChatCompletionsRoute, Error,
    common_utils::string_headers,
    openai::{OpenAiChatCall, PreparedCall},
    stream::ReportedUsage,
};

const ENV_REFERENCE: &str = "os.environ/";

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
            adapter: Some(ResponseAdapter { config, shape }),
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
    fn frame(&self, block: SseBlock, usage: &ReportedUsage) -> Option<Bytes> {
        if block.event.data.trim() == "[DONE]" {
            return Some(Bytes::from_static(DONE_FRAME));
        }
        let Ok(Value::Object(mut chunk)) = serde_json::from_str::<Value>(&block.event.data) else {
            return Some(block.bytes);
        };
        let mut changed = self.config.fix_chunk(&mut chunk, &self.shape);
        if let Some(reported) = chunk.get("usage").and_then(usage_from_json) {
            usage.set(reported);
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
        let mut frame = b"data: ".to_vec();
        serde_json::to_writer(&mut frame, &chunk).ok()?;
        frame.extend_from_slice(b"\n\n");
        Some(Bytes::from(frame))
    }
}

struct Relay {
    upstream: BoxStream<'static, Result<Bytes, Error>>,
    buffer: BytesMut,
    codec: SseBlockCodec,
    adapter: ResponseAdapter,
    usage: ReportedUsage,
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
                    if let Some(frame) = self.adapter.frame(block, &self.usage) {
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
