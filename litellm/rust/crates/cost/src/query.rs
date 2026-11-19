//! The typed query documents `price` answers. A caller builds a `PriceQuery` directly (the native Messages route) or
//! the bridge deserializes one from the facts Python read off its objects. Every kind shares one `Header`: the
//! candidates to price, where the prices come from, the call facts that pick a rate, and whether discount and margin
//! apply. Python-shaped interpretation (candidate order, the served tier, reported-cost precedence) happens before a
//! query is built, never here

use std::collections::BTreeMap;
use std::time::Duration;

use crate::call::{InferenceGeo, Speed};
use crate::money::UnitPrice;
use crate::price::{CacheReadMode, CacheTtl, Residency, SearchContextSize, ServiceTier, TokenKind};
use crate::row::RowRef;
use crate::usage::{Characters, DurationMeter, Images, Search, Tokens, Usage, WebSearch};

pub use crate::image::{ImageCalculator, ImageQuery};
pub use crate::ocr::OcrQuery;
pub use crate::rerank::RerankQuery;
pub use crate::search::SearchQuery;
pub use crate::speech::SpeechQuery;
pub use crate::transcription::TranscriptionQuery;
pub use crate::vector_store::VectorStoreSearchQuery;
pub use crate::video::VideoQuery;

/// One call to price, tagged by `kind`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "kind", rename_all = "snake_case")
)]
pub enum PriceQuery {
    Tokens(Box<TokensQuery>),
    Image(Box<ImageQuery>),
    Speech(Box<SpeechQuery>),
    Transcription(Box<TranscriptionQuery>),
    Ocr(Box<OcrQuery>),
    Rerank(Box<RerankQuery>),
    Search(Box<SearchQuery>),
    VectorStoreSearch(Box<VectorStoreSearchQuery>),
    Video(Box<VideoQuery>),
    /// Realtime, Live and Responses websocket sessions
    Session(Header),
    BatchLine(Header),
    Guardrail(Header),
    A2a(Header),
    Mcp(Header),
}

/// The kind of a query, named in `Unpriced::NotPorted`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::VariantArray, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "tokens")]
    Tokens,
    #[strum(serialize = "image")]
    Image,
    #[strum(serialize = "speech")]
    Speech,
    #[strum(serialize = "transcription")]
    Transcription,
    #[strum(serialize = "ocr")]
    Ocr,
    #[strum(serialize = "rerank")]
    Rerank,
    #[strum(serialize = "search")]
    Search,
    #[strum(serialize = "vector_store_search")]
    VectorStoreSearch,
    #[strum(serialize = "video")]
    Video,
    #[strum(serialize = "session")]
    Session,
    #[strum(serialize = "batch_line")]
    BatchLine,
    #[strum(serialize = "guardrail")]
    Guardrail,
    #[strum(serialize = "a2a")]
    A2a,
    #[strum(serialize = "mcp")]
    Mcp,
}

/// What every kind carries
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Header {
    /// Model names to price under, in the order main tries them
    pub candidates: Vec<Candidate>,
    pub source: PriceSource,
    pub basis: Basis,
    pub adjustments: AdjustmentPolicy,
    /// A cost the provider or a hook stated for the call
    #[cfg_attr(feature = "serde", serde(default))]
    pub reported: Option<ReportedCost>,
}

/// A model name and the provider whose rules price it
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Candidate {
    pub model: Box<str>,
    pub provider: Provider,
}

/// The provider whose pricing rules apply, `custom_llm_provider` as main passes it to `cost_per_token`
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, strum::EnumString, strum::Display)]
#[cfg_attr(feature = "serde", derive(serde_with::DeserializeFromStr))]
pub enum Provider {
    #[strum(serialize = "anthropic")]
    Anthropic,
    #[strum(serialize = "bedrock")]
    Bedrock,
    #[strum(serialize = "vertex_ai")]
    VertexAi,
    #[strum(serialize = "azure_ai")]
    AzureAi,
    #[strum(default, transparent)]
    Other(String),
}

/// Where the rates come from
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "from", rename_all = "snake_case")
)]
pub enum PriceSource {
    /// The first candidate the catalog has a row for
    Catalog,
    /// One row the caller resolved, priced under the first candidate's provider. `breakdown_by_name` is set for
    /// `requested_model_info_override`, whose stored breakdown main reads off the row the candidate names instead
    Row {
        row: RowRef,
        #[cfg_attr(feature = "serde", serde(default))]
        breakdown_by_name: bool,
    },
    /// `custom_cost_per_token`. A missing cache rate is the input rate
    FlatPerToken(FlatRates),
    /// `custom_cost_per_second`
    PerSecond { rate: UnitPrice },
    /// Rows the caller resolved, such as a deployment's own prices, read ahead of the first candidate's catalog row
    /// in the order the kind's main calculator reads them
    Rows { rows: Vec<RowRef> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct FlatRates {
    pub input: UnitPrice,
    pub output: UnitPrice,
    #[cfg_attr(feature = "serde", serde(default))]
    pub cache_read: Option<UnitPrice>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub cache_write: Option<UnitPrice>,
}

/// The call facts that pick a rate, echoed on the receipt
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct Basis {
    #[cfg_attr(feature = "serde", serde(default))]
    pub service_tier: ServiceTier,
    #[cfg_attr(feature = "serde", serde(default))]
    pub data_residency: Option<DataResidency>,
    /// The Vertex AI location that served the call
    #[cfg_attr(feature = "serde", serde(default))]
    pub vertex_location: Option<Box<str>>,
    /// When the call was billed, for off-peak windows
    pub at: jiff::Timestamp,
}

/// `data_residency`, lowercased by the caller
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum DataResidency {
    Eu,
    Us,
    Fedramp,
}

/// Whether the cost-map discount and margin apply. `completion_cost` applies them, `cost_per_token` does not
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "policy", rename_all = "snake_case")
)]
pub enum AdjustmentPolicy {
    None,
    Apply {
        /// `custom_llm_provider` of the call, which keys both the discount and the margin
        #[cfg_attr(feature = "serde", serde(default))]
        provider: Option<Box<str>>,
        /// The model the discount is keyed on when it is not the candidate that priced, as for custom pricing
        #[cfg_attr(feature = "serde", serde(default))]
        model: Option<Box<str>>,
        #[cfg_attr(feature = "serde", serde(default))]
        region: Option<Box<str>>,
    },
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ReportedCost {
    /// Which field stated it, such as `usage.cost`
    pub source: Box<str>,
    pub usd: f64,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct TokensQuery {
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub header: Header,
    pub usage: TokenUsage,
    /// The model the client asked for, which decides the Azure AI Model Router fee
    #[cfg_attr(feature = "serde", serde(default))]
    pub requested_model: Option<Box<str>>,
}

/// Billed quantities of a token call, with the overlap between cached and per-modality counts already removed
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct TokenUsage {
    /// Prompt size as reported, cache included. Only picks the band
    pub prompt_tokens: u64,
    /// Uncached prompt tokens
    pub input: Modalities,
    pub cache_read: CacheReads,
    pub cache_write: CacheWrites,
    /// Completion tokens, reasoning excluded
    pub output: Modalities,
    pub reasoning: u64,
    pub input_characters: u64,
    pub input_images: u64,
    pub input_audio_seconds: f64,
    pub input_video_seconds: f64,
    pub queries: u64,
    /// `server_tool_use.web_search_requests` when the usage reports it, zero included
    pub web_search: Option<WebSearchUse>,
    pub speed: Option<Speed>,
    pub inference_geo: Option<InferenceGeo>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct Modalities {
    pub text: u64,
    pub audio: u64,
    pub image: u64,
    pub video: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct CacheReads {
    pub text: u64,
    pub audio: u64,
    pub image: u64,
    /// The reads hit a cache the caller created
    pub explicit: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct CacheWrites {
    /// Every write not reported as 1h or 12h
    pub default: u64,
    pub one_hour: u64,
    pub twelve_hours: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct WebSearchUse {
    pub requests: u64,
    /// `web_search_options.search_context_size` of the request, read only by providers that price per call
    #[cfg_attr(feature = "serde", serde(default))]
    pub context: Option<SearchContextSize>,
}

impl PriceQuery {
    pub fn kind(&self) -> Kind {
        match self {
            Self::Tokens(_) => Kind::Tokens,
            Self::Image(_) => Kind::Image,
            Self::Speech(_) => Kind::Speech,
            Self::Transcription(_) => Kind::Transcription,
            Self::Ocr(_) => Kind::Ocr,
            Self::Rerank(_) => Kind::Rerank,
            Self::Search(_) => Kind::Search,
            Self::VectorStoreSearch(_) => Kind::VectorStoreSearch,
            Self::Video(_) => Kind::Video,
            Self::Session(_) => Kind::Session,
            Self::BatchLine(_) => Kind::BatchLine,
            Self::Guardrail(_) => Kind::Guardrail,
            Self::A2a(_) => Kind::A2a,
            Self::Mcp(_) => Kind::Mcp,
        }
    }
}

impl Basis {
    /// The residency uplift the call names
    pub fn residency(&self) -> Option<Residency> {
        match self.data_residency? {
            DataResidency::Eu => Some(Residency::Eu),
            DataResidency::Us => Some(Residency::Us),
            DataResidency::Fedramp => None,
        }
    }

    pub fn fedramp(&self) -> bool {
        self.data_residency == Some(DataResidency::Fedramp)
    }

    /// `get_vertex_regional_endpoint_uplift`: any location but `global`
    pub fn non_global_endpoint(&self) -> bool {
        self.vertex_location
            .as_deref()
            .is_some_and(|location| !location.eq_ignore_ascii_case("global"))
    }
}

impl TokenUsage {
    /// The quantities `calculate` bills
    pub fn billed(&self) -> Usage {
        let nonzero = |pairs: Vec<(TokenKind, u64)>| -> BTreeMap<TokenKind, u64> {
            pairs
                .into_iter()
                .filter(|(_, tokens)| *tokens > 0)
                .collect()
        };
        let read_mode = if self.cache_read.explicit {
            CacheReadMode::Explicit
        } else {
            CacheReadMode::Implicit
        };
        Usage {
            prompt_tokens: self.prompt_tokens,
            tokens: Tokens {
                input: nonzero(self.input.by_kind()),
                cache_read: [
                    ((TokenKind::Text, read_mode), self.cache_read.text),
                    ((TokenKind::Audio, read_mode), self.cache_read.audio),
                    ((TokenKind::Image, read_mode), self.cache_read.image),
                ]
                .into_iter()
                .filter(|(_, tokens)| *tokens > 0)
                .collect(),
                cache_write: [
                    (
                        (TokenKind::Text, CacheTtl::Default),
                        self.cache_write.default,
                    ),
                    (
                        (TokenKind::Text, CacheTtl::OneHour),
                        self.cache_write.one_hour,
                    ),
                    (
                        (TokenKind::Text, CacheTtl::TwelveHours),
                        self.cache_write.twelve_hours,
                    ),
                ]
                .into_iter()
                .filter(|(_, tokens)| *tokens > 0)
                .collect(),
                output: nonzero(self.output.by_kind()),
                reasoning: self.reasoning,
                citation: 0,
            },
            characters: Characters {
                prompt: self.input_characters,
                completion: 0,
            },
            durations: [
                (DurationMeter::InputAudio, self.input_audio_seconds),
                (DurationMeter::InputVideo, self.input_video_seconds),
            ]
            .into_iter()
            .filter(|(_, seconds)| *seconds > 0.0)
            .map(|(meter, seconds)| (meter, Duration::from_secs_f64(seconds)))
            .collect(),
            images: Images {
                input: self.input_images,
                ..Images::default()
            },
            search: Search {
                queries: self.queries,
                web_search: self.web_search.map(|used| WebSearch {
                    requests: used.requests,
                    context: used.context.unwrap_or(SearchContextSize::Medium),
                }),
                ..Search::default()
            },
            ..Usage::default()
        }
    }

    /// Every cache read, whatever its modality
    pub fn cache_read_total(&self) -> u64 {
        self.cache_read.text + self.cache_read.audio + self.cache_read.image
    }

    pub fn cache_write_total(&self) -> u64 {
        self.cache_write.default + self.cache_write.one_hour + self.cache_write.twelve_hours
    }

    /// Completion tokens, reasoning included
    pub fn completion_total(&self) -> u64 {
        self.output.total() + self.reasoning
    }
}

impl Modalities {
    fn by_kind(self) -> Vec<(TokenKind, u64)> {
        vec![
            (TokenKind::Text, self.text),
            (TokenKind::Audio, self.audio),
            (TokenKind::Image, self.image),
            (TokenKind::Video, self.video),
        ]
    }

    pub fn total(self) -> u64 {
        self.text + self.audio + self.image + self.video
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::anthropic("anthropic", Provider::Anthropic)]
    #[case::bedrock("bedrock", Provider::Bedrock)]
    #[case::vertex("vertex_ai", Provider::VertexAi)]
    #[case::azure_ai("azure_ai", Provider::AzureAi)]
    #[case::other("bedrock_mantle", Provider::Other("bedrock_mantle".to_owned()))]
    fn a_provider_parses_its_custom_llm_provider_spelling(
        #[case] written: &str,
        #[case] expected: Provider,
    ) {
        assert_eq!(written.parse::<Provider>(), Ok(expected.clone()));
        assert_eq!(expected.to_string(), written);
    }

    #[rstest]
    #[case::global(Some("GLOBAL"), false)]
    #[case::regional(Some("us-east5"), true)]
    #[case::none(None, false)]
    fn only_a_non_global_location_is_a_regional_endpoint(
        #[case] location: Option<&str>,
        #[case] expected: bool,
    ) {
        let basis = Basis {
            service_tier: ServiceTier::Standard,
            data_residency: None,
            vertex_location: location.map(Into::into),
            at: jiff::Timestamp::UNIX_EPOCH,
        };
        assert_eq!(basis.non_global_endpoint(), expected);
    }

    #[rstest]
    fn billed_usage_keys_explicit_reads_and_drops_empty_meters() {
        let usage = TokenUsage {
            prompt_tokens: 100,
            input: Modalities {
                text: 10,
                ..Modalities::default()
            },
            cache_read: CacheReads {
                text: 60,
                explicit: true,
                ..CacheReads::default()
            },
            cache_write: CacheWrites {
                one_hour: 30,
                ..CacheWrites::default()
            },
            ..TokenUsage::default()
        };
        let billed = usage.billed();

        assert_eq!(billed.tokens.input, BTreeMap::from([(TokenKind::Text, 10)]));
        assert_eq!(
            billed.tokens.cache_read,
            BTreeMap::from([((TokenKind::Text, CacheReadMode::Explicit), 60)])
        );
        assert_eq!(
            billed.tokens.cache_write,
            BTreeMap::from([((TokenKind::Text, CacheTtl::OneHour), 30)])
        );
        assert!(billed.durations.is_empty());
    }

    #[cfg(feature = "serde")]
    #[rstest]
    #[case::tokens(serde_json::json!({"kind": "tokens", "usage": {"prompt_tokens": 3}}), Kind::Tokens)]
    #[case::session(serde_json::json!({"kind": "session"}), Kind::Session)]
    #[case::vector_store(
        serde_json::json!({"kind": "vector_store_search", "queries": 1}),
        Kind::VectorStoreSearch
    )]
    fn a_query_deserializes_by_kind(#[case] body: serde_json::Value, #[case] expected: Kind) {
        let header = serde_json::json!({
            "candidates": [{"model": "claude-opus-4-8", "provider": "anthropic"}],
            "source": {"from": "catalog"},
            "basis": {"service_tier": "priority", "at": "2026-08-26T07:00:00Z"},
            "adjustments": {"policy": "apply", "provider": "anthropic"},
        });
        let merged: serde_json::Map<String, serde_json::Value> = header
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(body.as_object().unwrap().clone())
            .collect();
        let query: PriceQuery = serde_json::from_value(serde_json::Value::Object(merged)).unwrap();

        assert_eq!(query.kind(), expected);
        assert_eq!(<&'static str>::from(query.kind()), body["kind"]);
    }
}
