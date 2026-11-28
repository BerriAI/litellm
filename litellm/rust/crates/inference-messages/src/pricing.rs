//! Pricing a finished Messages call in Rust. The route reads the usage off the provider's answer (a message body, or
//! a stream's `message_start` and final `message_delta`), resolves the billable service tier as main's
//! `completion_cost` does, and hands the quantities with the deployment's pricing identity to a [`Pricing`]. A call
//! the route cannot price settles `Deferred`, and the host prices it the old way

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures_util::{StreamExt, stream::BoxStream};
use litellm_cost::adjust::Margin;
use litellm_cost::candidates::{AdjustFacts, CallIdentity, Named, PricingTable, TokenCall, priced};
use litellm_cost::query::TokenUsage;
use litellm_cost::receipt::Receipt;
use litellm_framer::sse::{SseCodec, SseEvent};
use litellm_host::interceptors::{Cost, PricedCost, StreamMeter};
use litellm_llms_types::formats::messages::{MessagesResponse, MessagesUsage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio_util::codec::Decoder;

use crate::Error;
use crate::usage::{Metered, UsageContext, metered};

/// The deployment facts main's `Logging.response_cost_calculator` prices a call with, sent by the host with the call.
/// A host sends none for a call main prices another way (custom pricing, a pricing provider override), and the route
/// then leaves it `Deferred`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PricingIdentity {
    /// The router deployment's `model_info.id`, which keys the row the router registered for it
    #[serde(default)]
    pub deployment_id: Option<String>,
    #[serde(default)]
    pub base_model: Option<String>,
    /// The Bedrock Mantle region the call was priced under
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub vertex_location: Option<String>,
    #[serde(default)]
    pub data_residency: Option<String>,
}

/// One finished call to price
#[derive(Clone, Debug, PartialEq)]
pub struct PricedCall {
    /// `custom_llm_provider` as the route resolved it
    pub provider: String,
    /// The model the route sent
    pub request_model: String,
    /// The model the provider's answer names
    pub response_model: Option<String>,
    pub identity: PricingIdentity,
    pub usage: TokenUsage,
    /// The billable tier, already resolved from the served and requested tiers
    pub service_tier: Option<String>,
}

/// Prices a finished call over the host's price table, `None` when no row prices it
pub trait Pricing: Send + Sync {
    fn price(&self, call: &PricedCall) -> Option<Receipt>;
}

impl PricedCall {
    /// The facts `completion_cost` reads off a native `/v1/messages` result on main: the route's provider and model,
    /// the model the response names and the deployment facts the proxy sent, discount and margin applied, no row of its
    /// own
    pub fn token_call(&self, at: jiff::Timestamp) -> TokenCall {
        let identity = &self.identity;
        TokenCall {
            named: Named::Call(CallIdentity {
                provider: self.provider.clone(),
                request_model: Some(self.request_model.clone()),
                response_model: self.response_model.clone(),
                base_model: identity.base_model.clone(),
                deployment_id: identity.deployment_id.clone(),
                region: identity.region.clone(),
                ..CallIdentity::default()
            }),
            usage: self.usage.clone(),
            requested_model: Some(self.request_model.as_str().into()),
            service_tier: self.service_tier.clone(),
            data_residency: identity.data_residency.clone(),
            vertex_location: identity.vertex_location.as_deref().map(Into::into),
            at,
            adjust: Some(AdjustFacts {
                provider: Some(self.provider.as_str().into()),
                model: None,
                region: identity.region.as_deref().map(Into::into),
            }),
        }
    }
}

/// Prices a finished Messages call over any catalog through litellm-cost's token-call entry point, the path a host
/// with no Python takes as well as the Python bridge
pub fn price_call<T: PricingTable + ?Sized>(
    table: &T,
    call: &PricedCall,
    at: jiff::Timestamp,
) -> Option<Receipt> {
    priced(table, call.token_call(at), None::<fn(&str) -> T::Row>)
}

/// The quantities a call is priced on. A `free` call costs nothing at all, the operator's fixed margin included
struct Billed {
    metered: Metered,
    free: bool,
}

/// The refusal categories Anthropic bills before any output
/// (<https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#how-refusals-are-billed>)
const REFUSALS_BILLED_BEFORE_OUTPUT: [&str; 3] = ["bio", "frontier_llm", "reasoning_extraction"];

/// How an answer stopped, as Anthropic's refusal billing reads it
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
struct Stop {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_details: Option<Value>,
}

impl Stop {
    /// Anthropic bills nothing for a refusal before any output outside the categories it names, whatever tokens the
    /// usage counts. A refusal after partial output is billed, so any output token keeps the charge. A call with
    /// more than one attempt is never free, since each attempt is billed by its own refusal category
    fn is_unbilled_refusal(&self, output_tokens: u64) -> bool {
        let category = self
            .stop_details
            .as_ref()
            .and_then(|details| details.get("category"))
            .and_then(Value::as_str);
        self.stop_reason.as_deref() == Some("refusal")
            && output_tokens == 0
            && !category.is_some_and(|category| REFUSALS_BILLED_BEFORE_OUTPUT.contains(&category))
    }
}

/// The facts one call's price reads beside its usage
pub(crate) struct CallPricing {
    pricing: Arc<dyn Pricing>,
    provider: String,
    request_model: String,
    identity: PricingIdentity,
    requested_tier: Option<String>,
    requested_fast: bool,
    inject_stream_cost: bool,
}

impl CallPricing {
    pub(crate) fn new(
        pricing: Arc<dyn Pricing>,
        identity: PricingIdentity,
        provider: String,
        request_model: String,
        requested: RequestedPricing,
    ) -> Self {
        Self {
            pricing,
            provider,
            request_model,
            identity,
            requested_tier: requested.service_tier,
            requested_fast: requested.fast,
            inject_stream_cost: requested.inject_stream_cost,
        }
    }

    fn priced(&self, billed: Billed, response_model: Option<String>) -> Option<PricedCost> {
        let Billed { metered, free } = billed;
        let service_tier = billable_tier(
            self.requested_tier.as_deref(),
            metered.served_tier.as_deref(),
        );
        let receipt = self.pricing.price(&PricedCall {
            provider: self.provider.clone(),
            request_model: self.request_model.clone(),
            response_model,
            identity: self.identity.clone(),
            usage: metered.usage,
            service_tier: service_tier.clone(),
        })?;
        let receipt = if free {
            Receipt {
                margin: Margin::default(),
                total: 0.0,
                ..receipt
            }
        } else {
            receipt
        };
        Some(PricedCost {
            receipt,
            service_tier,
            data_residency: self.identity.data_residency.clone(),
        })
    }

    /// The billed quantities of `usage` on an answer that stopped as `stop` says
    fn billed(&self, usage: &MessagesUsage, stop: &Stop, context: UsageContext) -> Option<Billed> {
        let metered = metered(usage, context)?;
        let output_tokens = metered.usage.output.text + metered.usage.reasoning;
        let attempts = usage.iterations.as_deref().map_or(0, <[_]>::len);
        if self.provider == "anthropic" && attempts <= 1 && stop.is_unbilled_refusal(output_tokens)
        {
            return Some(Billed {
                metered: Metered {
                    usage: TokenUsage::default(),
                    ..metered
                },
                free: true,
            });
        }
        Some(Billed {
            metered,
            free: false,
        })
    }

    /// The cost of a message body
    pub(crate) fn message(&self, message: &MessagesResponse) -> Cost {
        let thinking = message.content.iter().any(is_thinking_block);
        let stop = Stop {
            stop_reason: message.stop_reason.clone(),
            stop_details: message.extra.get("stop_details").cloned(),
        };
        message
            .usage
            .clone()
            .and_then(|usage| serde_json::from_value::<MessagesUsage>(usage).ok())
            .and_then(|usage| self.billed(&usage, &stop, self.context(thinking)))
            .and_then(|metered| self.priced(metered, Some(message.model.clone())))
            .map_or(Cost::Deferred, |priced| Cost::Priced(Box::new(priced)))
    }

    fn context(&self, thinking: bool) -> UsageContext {
        UsageContext {
            requested_fast: self.requested_fast,
            thinking,
        }
    }

    /// Meters a stream: its chunks pass through unchanged, except the usage events that get their cost injected when
    /// the host asked for it, and `meter` settles once the stream ends with its usage final
    pub(crate) fn metered_stream(
        self: Arc<Self>,
        meter: StreamMeter,
        chunks: BoxStream<'static, Result<Bytes, Error>>,
    ) -> BoxStream<'static, Result<Bytes, Error>> {
        let watch = StreamWatch {
            pricing: self,
            meter,
            pending: BytesMut::new(),
            codec: SseCodec::default(),
            usage: Map::new(),
            response_model: None,
            thinking: false,
            stop: Stop::default(),
            delta_last: false,
        };
        futures_util::stream::unfold((chunks, Some(watch)), |(mut chunks, watch)| async move {
            let mut watch = watch?;
            match chunks.next().await {
                Some(Ok(chunk)) => {
                    let chunk = watch.observe(chunk);
                    Some((Ok(chunk), (chunks, Some(watch))))
                }
                Some(Err(error)) => Some((Err(error), (chunks, None))),
                None => {
                    watch.settle();
                    None
                }
            }
        })
        .boxed()
    }
}

/// What the request asked for that the price reads
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RequestedPricing {
    pub(crate) service_tier: Option<String>,
    pub(crate) fast: bool,
    pub(crate) inject_stream_cost: bool,
}

/// The service tier a call bills (main's `_resolve_billable_service_tier`, which Python now calls): a served tier
/// that names a priced tier wins, one that names base pricing means standard, and otherwise the request decides, `auto`
/// meaning standard
pub fn billable_tier(requested: Option<&str>, served: Option<&str>) -> Option<String> {
    const PRICED: [&str; 5] = ["flex", "balanced", "priority", "fast", "ultrafast"];
    const BASE: [&str; 2] = ["default", "standard"];
    let served = served.map(str::to_lowercase);
    match served.as_deref() {
        Some(tier) if PRICED.contains(&tier) => served,
        Some(tier) if BASE.contains(&tier) => None,
        _ => requested
            .filter(|tier| !tier.eq_ignore_ascii_case("auto"))
            .map(str::to_owned),
    }
}

fn is_thinking_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    )
}

/// The Anthropic stream events the meter reads
#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StartedMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart { content_block: Value },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {},
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {},
    #[serde(rename = "message_delta")]
    MessageDelta {
        #[serde(default)]
        delta: Stop,
        #[serde(default)]
        usage: Option<Map<String, Value>>,
    },
    #[serde(rename = "message_stop")]
    MessageStop {},
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct StartedMessage {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<Map<String, Value>>,
}

/// One stream's usage as the events go by. Like Python's `_handle_usage`, each event's usage object is merged over the
/// ones before it, key by key
struct StreamWatch {
    pricing: Arc<CallPricing>,
    meter: StreamMeter,
    pending: BytesMut,
    codec: SseCodec,
    usage: Map<String, Value>,
    response_model: Option<String>,
    thinking: bool,
    stop: Stop,
    /// The last event of the stream's terminal region was a `message_delta`. Main re-tokenizes a stream that ended
    /// anywhere else (`_stream_was_interrupted`), so only such a stream is priced here
    delta_last: bool,
}

impl StreamWatch {
    /// Decodes each event's JSON once. A chunk that cannot carry an injected cost reads the typed event straight from
    /// the text, which skips the content it does not name. One that can parses each event into a `Value` that both the
    /// meter and the injection read
    fn observe(&mut self, chunk: Bytes) -> Bytes {
        self.pending.extend_from_slice(&chunk);
        let events: Vec<SseEvent> =
            std::iter::from_fn(|| self.codec.decode(&mut self.pending).ok().flatten()).collect();
        if !(self.pricing.inject_stream_cost && ends_on_a_frame(&chunk)) {
            events
                .iter()
                .filter_map(|event| serde_json::from_str::<StreamEvent>(&event.data).ok())
                .for_each(|event| self.read(event));
            return chunk;
        }
        let parsed: Vec<(String, Value)> = events
            .into_iter()
            .filter_map(|event| {
                let value = serde_json::from_str::<Value>(&event.data).ok()?;
                if let Ok(typed) = StreamEvent::deserialize(&value) {
                    self.read(typed);
                }
                Some((event.data, value))
            })
            .collect();
        self.with_injected_cost(chunk, &parsed)
    }

    fn read(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => {
                self.response_model = message.model;
                self.merge(message.usage);
                self.delta_last = false;
            }
            StreamEvent::ContentBlockStart { content_block } => {
                self.thinking |= is_thinking_block(&content_block);
            }
            StreamEvent::ContentBlockDelta {} | StreamEvent::ContentBlockStop {} => {
                self.delta_last = false;
            }
            StreamEvent::MessageDelta { delta, usage } => {
                self.stop = delta;
                self.merge(usage);
                self.delta_last = true;
            }
            StreamEvent::MessageStop {} => self.settle(),
            StreamEvent::Other => {}
        }
    }

    fn merge(&mut self, usage: Option<Map<String, Value>>) {
        self.usage.extend(usage.into_iter().flatten());
    }

    /// Prices the stream's merged usage at its `message_stop`, which a caller may stop reading at, or else where the
    /// stream ends. Main's assembled stream response drops the usage's `service_tier`, so the served tier never reaches
    /// its price and the request alone decides it
    fn settle(&mut self) {
        if !self.delta_last || self.meter.settled().is_some() {
            return;
        }
        let priced = serde_json::from_value::<MessagesUsage>(Value::Object(self.usage.clone()))
            .ok()
            .and_then(|usage| {
                self.pricing
                    .billed(&usage, &self.stop, self.pricing.context(self.thinking))
            })
            .and_then(|billed| {
                self.pricing.priced(
                    Billed {
                        metered: Metered {
                            served_tier: None,
                            ..billed.metered
                        },
                        ..billed
                    },
                    self.response_model.clone(),
                )
            });
        if let Some(priced) = priced {
            self.meter.settle(priced);
        }
    }

    /// `process_chunk_with_cost_injection` for a chunk of whole SSE frames: the first `message_delta` line whose own
    /// usage prices gets that cost as `usage.cost`, written back the way Python's `json.dumps` writes it. Each line is
    /// matched to the event `observe` already parsed, so nothing is decoded twice
    fn with_injected_cost(&self, chunk: Bytes, parsed: &[(String, Value)]) -> Bytes {
        let Ok(text) = std::str::from_utf8(&chunk) else {
            return chunk;
        };
        let lines: Vec<&str> = text.split('\n').collect();
        let injected = lines.iter().enumerate().find_map(|(index, line)| {
            let data = line.trim().strip_prefix("data:")?.trim();
            let (_, event) = parsed.iter().find(|(raw, _)| raw.trim() == data)?;
            let rewritten = self.priced_event(event)?;
            let carriage = if line.ends_with('\r') { "\r" } else { "" };
            Some((index, format!("data: {rewritten}{carriage}")))
        });
        let Some((index, rewritten)) = injected else {
            return chunk;
        };
        let joined: Vec<&str> = lines
            .iter()
            .enumerate()
            .map(|(at, line)| {
                if at == index {
                    rewritten.as_str()
                } else {
                    line
                }
            })
            .collect();
        Bytes::from(joined.join("\n"))
    }

    /// The event with its usage's cost, for a `message_delta` whose own usage prices. Main reads that usage alone,
    /// without the request's speed or the stream's thinking, and names the requested model as the response's
    fn priced_event(&self, event: &Value) -> Option<String> {
        let Value::Object(fields) = event else {
            return None;
        };
        if fields.get("type").and_then(Value::as_str) != Some("message_delta") {
            return None;
        }
        let usage_value = fields.get("usage")?;
        let Value::Object(usage) = usage_value else {
            return None;
        };
        let read = MessagesUsage::deserialize(usage_value).ok()?;
        let stop = fields
            .get("delta")
            .and_then(|delta| Stop::deserialize(delta).ok())
            .unwrap_or_default();
        let priced = self
            .pricing
            .billed(&read, &stop, UsageContext::default())
            .and_then(|metered| {
                self.pricing
                    .priced(metered, Some(self.pricing.request_model.clone()))
            })?;
        let cost = serde_json::Number::from_f64(priced.receipt.total)?;
        let usage: Map<String, Value> = usage
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([("cost".to_owned(), Value::Number(cost))])
            .collect();
        let event: Map<String, Value> = fields
            .iter()
            .map(|(key, value)| match key.as_str() {
                "usage" => (key.clone(), Value::Object(usage.clone())),
                _ => (key.clone(), value.clone()),
            })
            .collect();
        litellm_python_compat::json::dumps(&litellm_python_compat::json::from_json(Value::Object(
            event,
        )))
        .ok()
    }
}

fn ends_on_a_frame(chunk: &[u8]) -> bool {
    chunk.ends_with(b"\n\n") || chunk.ends_with(b"\r\n\r\n")
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::TryStreamExt;
    use rstest::{fixture, rstest};

    use super::*;

    /// Captured once from main's Python with `LITELLM_LOCAL_MODEL_COST_MAP=True`: each provider's message body or SSE
    /// stream, and the billed usage, served tier and response model main priced it with (`null` where main does not
    /// take the token path the route ports)
    const PARITY: &str = include_str!("../fixtures/usage_parity.json");

    #[derive(Deserialize)]
    struct Case {
        name: String,
        provider: String,
        #[serde(default)]
        requested_speed: Option<String>,
        #[serde(default)]
        body: Option<MessagesResponse>,
        #[serde(default)]
        sse: Option<String>,
        expected: Option<Expected>,
    }

    #[derive(Deserialize)]
    struct Expected {
        usage: TokenUsage,
        served_tier: Option<String>,
        response_model: String,
    }

    fn case(name: &str) -> Case {
        let cases: Vec<Case> = serde_json::from_str(PARITY).unwrap();
        cases.into_iter().find(|case| case.name == name).unwrap()
    }

    /// Prices every call at one micro-dollar per prompt token and two per output token, and keeps what it was asked
    #[derive(Default)]
    struct Recording(Mutex<Vec<PricedCall>>);

    fn total(usage: &TokenUsage) -> f64 {
        (usage.prompt_tokens as f64 + 2.0 * usage.output.text as f64) / 1e6
    }

    impl Pricing for Recording {
        fn price(&self, call: &PricedCall) -> Option<Receipt> {
            self.0.lock().unwrap().push(call.clone());
            Some(Receipt::of_total(total(&call.usage)))
        }
    }

    impl Recording {
        fn calls(&self) -> Vec<PricedCall> {
            self.0.lock().unwrap().clone()
        }
    }

    #[fixture]
    fn recording() -> Arc<Recording> {
        Arc::default()
    }

    fn call_pricing(
        recording: &Arc<Recording>,
        case: &Case,
        requested: RequestedPricing,
    ) -> Arc<CallPricing> {
        Arc::new(CallPricing::new(
            recording.clone(),
            PricingIdentity {
                deployment_id: Some("deployment-1".into()),
                ..PricingIdentity::default()
            },
            case.provider.clone(),
            "requested-model".into(),
            RequestedPricing {
                fast: case.requested_speed.as_deref() == Some("fast"),
                ..requested
            },
        ))
    }

    /// The stream as a client reads it, cut into pieces that split events and lines
    async fn read(pricing: Arc<CallPricing>, meter: StreamMeter, sse: &str) -> Vec<Bytes> {
        let pieces: Vec<Result<Bytes, Error>> = sse
            .as_bytes()
            .chunks(7)
            .map(|piece| Ok(Bytes::copy_from_slice(piece)))
            .collect();
        pricing
            .metered_stream(meter, futures_util::stream::iter(pieces).boxed())
            .try_collect()
            .await
            .unwrap()
    }

    #[rstest]
    #[case::anthropic("anthropic_plain")]
    #[case::anthropic_cache_ttl_fast_geo_priority("anthropic_cache_ttl_fast_geo_priority")]
    #[case::anthropic_requested_fast("anthropic_requested_fast")]
    #[case::anthropic_iterations("anthropic_iterations")]
    #[case::anthropic_thinking_without_counts("anthropic_thinking_without_counts")]
    #[case::anthropic_web_search("anthropic_web_search")]
    #[case::bedrock_invoke("bedrock_invoke")]
    #[case::bedrock_claude_platform_geo("bedrock_claude_platform_geo")]
    #[case::bedrock_mantle("bedrock_mantle")]
    #[case::vertex_ai("vertex_ai")]
    #[case::azure_ai("azure_ai")]
    fn a_message_body_is_priced_on_the_usage_main_billed(
        recording: Arc<Recording>,
        #[case] name: &str,
    ) {
        let case = case(name);
        let pricing = call_pricing(&recording, &case, RequestedPricing::default());

        let cost = pricing.message(case.body.as_ref().unwrap());

        let Some(expected) = case.expected else {
            assert_eq!(cost, Cost::Deferred);
            assert!(recording.calls().is_empty());
            return;
        };
        let [call]: [PricedCall; 1] = recording.calls().try_into().unwrap();
        assert_eq!(call.usage, expected.usage);
        assert_eq!(
            call.response_model.as_deref(),
            Some(expected.response_model.as_str())
        );
        assert_eq!(
            call.service_tier,
            billable_tier(None, expected.served_tier.as_deref())
        );
        assert_eq!(
            (call.provider.as_str(), call.request_model.as_str()),
            (case.provider.as_str(), "requested-model")
        );
        let Cost::Priced(priced) = cost else {
            panic!("a body with usage the route reads is priced, got {cost:?}");
        };
        assert_eq!(priced.receipt.total, total(&expected.usage));
        assert_eq!(priced.service_tier, call.service_tier);
    }

    #[rstest]
    #[case::anthropic("anthropic_stream")]
    #[case::anthropic_cumulative_delta("anthropic_stream_cumulative_delta")]
    #[case::anthropic_requested_fast("anthropic_stream_requested_fast")]
    #[case::anthropic_geo_speed_tier("anthropic_stream_geo_speed_tier")]
    #[case::anthropic_thinking("anthropic_stream_thinking")]
    #[case::anthropic_interrupted("anthropic_stream_interrupted")]
    #[case::bedrock_invoke_metrics_folded("bedrock_invoke_stream_metrics_folded")]
    #[case::bedrock_mantle("bedrock_mantle_stream")]
    #[case::vertex_ai("vertex_ai_stream")]
    #[case::azure_ai("azure_ai_stream")]
    #[tokio::test]
    async fn a_stream_settles_its_meter_on_the_usage_main_billed(
        recording: Arc<Recording>,
        #[case] name: &str,
        #[values(false, true)] inject_stream_cost: bool,
    ) {
        let case = case(name);
        let sse = case.sse.clone().unwrap();
        let meter = StreamMeter::default();

        let delivered = read(
            call_pricing(
                &recording,
                &case,
                RequestedPricing {
                    inject_stream_cost,
                    ..RequestedPricing::default()
                },
            ),
            meter.clone(),
            &sse,
        )
        .await;

        assert_eq!(delivered.concat(), sse.as_bytes());
        let Some(expected) = case.expected else {
            assert_eq!(meter.settled(), None);
            return;
        };
        let [call]: [PricedCall; 1] = recording.calls().try_into().unwrap();
        assert_eq!(call.usage, expected.usage);
        assert_eq!(
            call.response_model.as_deref(),
            Some(expected.response_model.as_str())
        );
        assert_eq!(call.service_tier, None);
        assert_eq!(
            meter.settled().unwrap().receipt.total,
            total(&expected.usage)
        );
    }

    fn refusal_case(provider: &str) -> Case {
        Case {
            name: "refusal".into(),
            provider: provider.into(),
            requested_speed: None,
            body: None,
            sse: None,
            expected: None,
        }
    }

    fn refusal_body(stop_details: &Value, output_tokens: u64) -> MessagesResponse {
        serde_json::from_value(serde_json::json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-fable-5-1",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": null,
            "stop_details": stop_details,
            "usage": {
                "input_tokens": 13,
                "cache_creation_input_tokens": 6179,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6179},
                "output_tokens": output_tokens,
            },
        }))
        .unwrap()
    }

    fn refusal_details(category: Option<&str>) -> Value {
        serde_json::json!({"type": "refusal", "category": category, "explanation": null})
    }

    #[rstest]
    #[case::cyber("anthropic", refusal_details(Some("cyber")), 0, false)]
    #[case::general_harms("anthropic", refusal_details(Some("general_harms")), 0, false)]
    #[case::no_named_category("anthropic", refusal_details(None), 0, false)]
    #[case::no_stop_details("anthropic", Value::Null, 0, false)]
    #[case::bio("anthropic", refusal_details(Some("bio")), 0, true)]
    #[case::frontier_llm("anthropic", refusal_details(Some("frontier_llm")), 0, true)]
    #[case::reasoning_extraction(
        "anthropic",
        refusal_details(Some("reasoning_extraction")),
        0,
        true
    )]
    #[case::after_partial_output("anthropic", refusal_details(Some("cyber")), 7, true)]
    #[case::another_platform("bedrock", refusal_details(Some("cyber")), 0, true)]
    fn a_refusal_is_charged_only_where_anthropic_bills_it(
        recording: Arc<Recording>,
        #[case] provider: &str,
        #[case] stop_details: Value,
        #[case] output_tokens: u64,
        #[case] billed: bool,
    ) {
        let pricing = call_pricing(
            &recording,
            &refusal_case(provider),
            RequestedPricing::default(),
        );

        let cost = pricing.message(&refusal_body(&stop_details, output_tokens));

        let Cost::Priced(priced) = cost else {
            panic!("a refusal with usage is priced, got {cost:?}");
        };
        assert_eq!(priced.receipt.total > 0.0, billed);
    }

    #[rstest]
    #[case::an_earlier_attempt_with_output(7)]
    #[case::an_earlier_attempt_without_output(0)]
    fn a_refusal_that_followed_another_attempt_is_charged(
        recording: Arc<Recording>,
        #[case] earlier_output: u64,
    ) {
        let pricing = call_pricing(
            &recording,
            &refusal_case("anthropic"),
            RequestedPricing::default(),
        );
        let body: MessagesResponse = serde_json::from_value(serde_json::json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-fable-5-1",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": null,
            "stop_details": refusal_details(Some("cyber")),
            "usage": {
                "input_tokens": 13,
                "output_tokens": 0,
                "iterations": [
                    {"type": "message", "input_tokens": 13, "output_tokens": earlier_output},
                    {"type": "fallback_message", "input_tokens": 13, "output_tokens": 0},
                ],
            },
        }))
        .unwrap();

        let Cost::Priced(priced) = pricing.message(&body) else {
            panic!("a refusal with usage is priced");
        };

        assert!(priced.receipt.total > 0.0);
    }

    /// Prices as [`Recording`] does, plus the fixed fee an operator's `fixed_amount` margin adds to every call
    struct FixedFee;

    impl Pricing for FixedFee {
        fn price(&self, call: &PricedCall) -> Option<Receipt> {
            let priced = Receipt::of_total(total(&call.usage));
            Some(Receipt {
                margin: Margin {
                    percent: 0.0,
                    fixed: 0.5,
                    total: 0.5,
                },
                total: priced.total + 0.5,
                ..priced
            })
        }
    }

    #[rstest]
    #[case::unbilled(refusal_details(Some("cyber")), 0.0)]
    #[case::billed(refusal_details(Some("bio")), 0.5)]
    fn an_unbilled_refusal_carries_no_fixed_margin(
        #[case] stop_details: Value,
        #[case] margin: f64,
    ) {
        let pricing = CallPricing::new(
            Arc::new(FixedFee),
            PricingIdentity::default(),
            "anthropic".into(),
            "requested-model".into(),
            RequestedPricing::default(),
        );

        let cost = pricing.message(&refusal_body(&stop_details, 0));

        let Cost::Priced(priced) = cost else {
            panic!("a refusal with usage is priced, got {cost:?}");
        };
        assert_eq!(priced.receipt.margin.total, margin);
        assert_eq!(priced.receipt.total > 0.0, margin > 0.0);
    }

    fn refusal_stream(category: &str) -> String {
        format!(
            "event: message_start\ndata: {{\"type\": \"message_start\", \"message\": {{\"model\": \
             \"claude-fable-5-1\", \"usage\": {{\"input_tokens\": 13, \"cache_creation_input_tokens\": 6179, \
             \"output_tokens\": 0}}}}}}\n\nevent: message_delta\ndata: {{\"type\": \"message_delta\", \"delta\": \
             {{\"stop_reason\": \"refusal\", \"stop_sequence\": null, \"stop_details\": {{\"type\": \"refusal\", \
             \"category\": \"{category}\", \"explanation\": null}}}}, \"usage\": {{\"input_tokens\": 13, \
             \"cache_creation_input_tokens\": 6179, \"output_tokens\": 0}}}}\n\nevent: message_stop\ndata: \
             {{\"type\": \"message_stop\"}}\n\n"
        )
    }

    #[rstest]
    #[case::cyber("cyber", false)]
    #[case::bio("bio", true)]
    #[tokio::test]
    async fn a_refused_stream_is_charged_only_where_anthropic_bills_it(
        recording: Arc<Recording>,
        #[case] category: &str,
        #[case] billed: bool,
    ) {
        let pricing = call_pricing(
            &recording,
            &refusal_case("anthropic"),
            RequestedPricing {
                inject_stream_cost: true,
                ..RequestedPricing::default()
            },
        );
        let meter = StreamMeter::default();
        let frames: Vec<Result<Bytes, Error>> = refusal_stream(category)
            .split_inclusive("\n\n")
            .map(|frame| Ok(Bytes::from(frame.to_owned())))
            .collect();

        let delivered: Vec<Bytes> = pricing
            .metered_stream(meter.clone(), futures_util::stream::iter(frames).boxed())
            .try_collect()
            .await
            .unwrap();

        assert_eq!(meter.settled().unwrap().receipt.total > 0.0, billed);
        let injected: Value = delivered
            .iter()
            .map(|frame| std::str::from_utf8(frame).unwrap())
            .find(|frame| frame.contains("\"cost\""))
            .and_then(|frame| frame.lines().find_map(|line| line.strip_prefix("data: ")))
            .map(|data| serde_json::from_str(data).unwrap())
            .unwrap();
        assert_eq!(injected["usage"]["cost"].as_f64().unwrap() > 0.0, billed);
    }

    #[rstest]
    #[case::the_request_decides_when_the_usage_is_silent(Some("priority"), None, Some("priority"))]
    #[case::a_served_priced_tier_wins(Some("flex"), Some("Priority"), Some("priority"))]
    #[case::a_served_base_tier_means_standard(Some("priority"), Some("standard"), None)]
    #[case::auto_means_standard(Some("AUTO"), None, None)]
    #[case::an_unknown_served_tier_leaves_it_to_the_request(
        Some("flex"),
        Some("batch"),
        Some("flex")
    )]
    fn the_billable_tier_follows_main(
        #[case] requested: Option<&str>,
        #[case] served: Option<&str>,
        #[case] billed: Option<&str>,
    ) {
        assert_eq!(billable_tier(requested, served).as_deref(), billed);
    }

    struct Declining;

    impl Pricing for Declining {
        fn price(&self, _call: &PricedCall) -> Option<Receipt> {
            None
        }
    }

    /// The proxy's Python injector prices every usage event that reaches it without `usage.cost`, so an event the
    /// route cannot price must reach it untouched
    #[rstest]
    #[case::web_search("anthropic_stream", Arc::new(Recording::default()) as Arc<dyn Pricing>, true)]
    #[case::declined_bedrock_row("bedrock_invoke_stream_metrics_folded", Arc::new(Declining) as Arc<dyn Pricing>, false)]
    #[tokio::test]
    async fn a_usage_event_the_route_cannot_price_passes_through_for_python_to_price(
        #[case] name: &str,
        #[case] pricing: Arc<dyn Pricing>,
        #[case] web_search: bool,
    ) {
        let case = case(name);
        let sse = case.sse.unwrap();
        let sse = match web_search {
            true => sse.replace(
                "\"usage\": {\"output_tokens\": 120}",
                "\"usage\": {\"output_tokens\": 120, \"server_tool_use\": {\"web_search_requests\": 2}}",
            ),
            false => sse,
        };
        assert!(sse.contains("web_search_requests\": 2") == web_search);
        let call = Arc::new(CallPricing::new(
            pricing,
            PricingIdentity::default(),
            case.provider.clone(),
            "requested-model".into(),
            RequestedPricing {
                inject_stream_cost: true,
                ..RequestedPricing::default()
            },
        ));
        let frames: Vec<Result<Bytes, Error>> = sse
            .split_inclusive("\n\n")
            .map(|frame| Ok(Bytes::from(frame.to_owned())))
            .collect();

        let delivered: Vec<Bytes> = call
            .metered_stream(
                StreamMeter::default(),
                futures_util::stream::iter(frames).boxed(),
            )
            .try_collect()
            .await
            .unwrap();

        assert_eq!(delivered.concat(), sse.as_bytes());
    }

    #[rstest]
    #[tokio::test]
    async fn an_injected_usage_event_carries_the_cost_of_its_own_usage_in_python_json(
        recording: Arc<Recording>,
    ) {
        let case = case("anthropic_stream");
        let pricing = call_pricing(
            &recording,
            &case,
            RequestedPricing {
                inject_stream_cost: true,
                ..RequestedPricing::default()
            },
        );
        let frames: Vec<Result<Bytes, Error>> = case
            .sse
            .unwrap()
            .split_inclusive("\n\n")
            .map(|frame| Ok(Bytes::from(frame.to_owned())))
            .collect();

        let delivered: Vec<Bytes> = pricing
            .metered_stream(
                StreamMeter::default(),
                futures_util::stream::iter(frames).boxed(),
            )
            .try_collect()
            .await
            .unwrap();

        let frames: Vec<&str> = delivered
            .iter()
            .map(|frame| std::str::from_utf8(frame).unwrap())
            .collect();
        let [delta]: [&str; 1] = frames
            .iter()
            .copied()
            .filter(|frame| frame.contains("\"cost\""))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        assert_eq!(
            delta,
            "event: message_delta\ndata: {\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\", \
             \"stop_sequence\": null}, \"usage\": {\"output_tokens\": 120, \"cost\": 0.00024}}\n\n"
        );
        let [own, whole]: [PricedCall; 2] = recording.calls().try_into().unwrap();
        assert_eq!(own.usage.prompt_tokens, 0);
        assert_eq!(whole.usage.prompt_tokens, 1500);
    }

    #[rstest]
    #[tokio::test]
    async fn a_caller_that_stops_reading_at_message_stop_still_finds_the_price(
        recording: Arc<Recording>,
    ) {
        let case = case("anthropic_stream");
        let frames: Vec<Result<Bytes, Error>> = case
            .sse
            .clone()
            .unwrap()
            .split_inclusive("\n\n")
            .map(|frame| Ok(Bytes::from(frame.to_owned())))
            .collect();
        let count = frames.len();
        let meter = StreamMeter::default();
        let chunks = call_pricing(&recording, &case, RequestedPricing::default())
            .metered_stream(meter.clone(), futures_util::stream::iter(frames).boxed());

        let read: Vec<Bytes> = chunks.take(count).try_collect().await.unwrap();

        assert!(read.last().unwrap().starts_with(b"event: message_stop"));
        assert_eq!(
            meter.settled().map(|priced| priced.receipt.total),
            Some(total(&case.expected.unwrap().usage))
        );
        assert_eq!(recording.calls().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_that_fails_before_message_stop_never_settles(recording: Arc<Recording>) {
        let case = case("anthropic_stream");
        let sse = case.sse.clone().unwrap();
        let (before_stop, _) = sse.split_once("event: message_stop").unwrap();
        let meter = StreamMeter::default();
        let pieces = vec![
            Ok(Bytes::from(before_stop.to_owned())),
            Err(Error::InvalidProvider("cut".into())),
        ];

        let delivered: Vec<Result<Bytes, Error>> =
            call_pricing(&recording, &case, RequestedPricing::default())
                .metered_stream(meter.clone(), futures_util::stream::iter(pieces).boxed())
                .collect()
                .await;

        assert!(delivered.last().unwrap().is_err());
        assert_eq!(meter.settled(), None);
        assert!(recording.calls().is_empty());
    }
}
