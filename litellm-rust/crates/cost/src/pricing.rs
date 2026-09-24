use std::collections::BTreeMap;

use crate::error::CostError;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Rate {
    #[default]
    Missing,
    Null,
    Invalid,
    Value(f64),
}

impl Rate {
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(value),
            Self::Missing | Self::Null | Self::Invalid => None,
        }
    }

    pub fn checked(self) -> Result<f64, CostError> {
        match self {
            Self::Value(value) if value.is_finite() && value >= 0.0 => Ok(value),
            _ => Err(CostError::InvalidRate),
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[strum(serialize_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Flex,
    Priority,
    Fast,
    Ultrafast,
}

impl ServiceTier {
    pub const SUFFIXES: [Self; 5] = [
        Self::Ultrafast,
        Self::Priority,
        Self::Auto,
        Self::Flex,
        Self::Fast,
    ];

    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    strum::Display,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum Metric {
    #[strum(serialize = "annotation_cost_per_page")]
    AnnotationPerPage,
    #[strum(serialize = "cache_creation_input_audio_token_cost")]
    CacheCreationAudioToken,
    #[strum(serialize = "cache_creation_input_token_cost")]
    CacheCreationToken,
    #[strum(serialize = "cache_creation_input_token_cost_above_1hr")]
    CacheCreationToken1hr,
    #[strum(serialize = "cache_read_input_audio_token_cost")]
    CacheReadAudioToken,
    #[strum(serialize = "cache_read_input_image_token_cost")]
    CacheReadImageToken,
    #[strum(serialize = "cache_read_input_token_cost")]
    CacheReadToken,
    #[strum(serialize = "citation_cost_per_token")]
    CitationToken,
    #[strum(serialize = "code_interpreter_cost_per_session")]
    CodeInterpreterPerSession,
    #[strum(serialize = "computer_use_input_cost_per_1k_tokens")]
    ComputerUseInput1kTokens,
    #[strum(serialize = "computer_use_output_cost_per_1k_tokens")]
    ComputerUseOutput1kTokens,
    #[strum(serialize = "file_search_cost_per_1k_calls")]
    FileSearchPer1kCalls,
    #[strum(serialize = "file_search_cost_per_gb_per_day")]
    FileSearchPerGbPerDay,
    #[strum(serialize = "google_maps_grounding_cost_per_query")]
    GoogleMapsGroundingPerQuery,
    #[strum(serialize = "input_cost_per_audio_per_second")]
    InputAudioPerSecond,
    #[strum(serialize = "input_cost_per_audio_token")]
    InputAudioToken,
    #[strum(serialize = "input_cost_per_character")]
    InputCharacter,
    #[strum(serialize = "input_dbu_cost_per_token")]
    InputDbuToken,
    #[strum(serialize = "input_cost_per_image")]
    InputPerImage,
    #[strum(serialize = "input_cost_per_image_token")]
    InputPerImageToken,
    #[strum(serialize = "input_cost_per_pixel")]
    InputPerPixel,
    #[strum(serialize = "input_cost_per_query")]
    InputPerQuery,
    #[strum(serialize = "input_cost_per_request")]
    InputPerRequest,
    #[strum(serialize = "input_cost_per_second")]
    InputPerSecond,
    #[strum(serialize = "input_cost_per_token")]
    InputPerToken,
    #[strum(serialize = "input_cost_per_token_cache_hit")]
    InputPerTokenCacheHit,
    #[strum(serialize = "input_cost_per_video_per_second")]
    InputVideoPerSecond,
    #[strum(serialize = "input_cost_per_video_per_second_above_15s_interval")]
    InputVideoPerSecond15sInterval,
    #[strum(serialize = "input_cost_per_video_per_second_above_8s_interval")]
    InputVideoPerSecond8sInterval,
    #[strum(serialize = "input_cost_per_video_token")]
    InputVideoToken,
    #[strum(serialize = "ocr_cost_per_credit")]
    OcrPerCredit,
    #[strum(serialize = "ocr_cost_per_page")]
    OcrPerPage,
    #[strum(serialize = "output_cost_per_audio_token")]
    OutputAudioToken,
    #[strum(serialize = "output_cost_per_character")]
    OutputCharacter,
    #[strum(serialize = "output_dbu_cost_per_token")]
    OutputDbuToken,
    #[strum(serialize = "output_cost_per_image")]
    OutputPerImage,
    #[strum(serialize = "output_cost_per_image_1024")]
    OutputPerImage1024,
    #[strum(serialize = "output_cost_per_image_1536")]
    OutputPerImage1536,
    #[strum(serialize = "output_cost_per_image_512")]
    OutputPerImage512,
    #[strum(serialize = "output_cost_per_image_token")]
    OutputPerImageToken,
    #[strum(serialize = "output_cost_per_pixel")]
    OutputPerPixel,
    #[strum(serialize = "output_cost_per_second")]
    OutputPerSecond,
    #[strum(serialize = "output_cost_per_second_1080p")]
    OutputPerSecond1080p,
    #[strum(serialize = "output_cost_per_second_2k")]
    OutputPerSecond2k,
    #[strum(serialize = "output_cost_per_second_480p")]
    OutputPerSecond480p,
    #[strum(serialize = "output_cost_per_second_4k")]
    OutputPerSecond4k,
    #[strum(serialize = "output_cost_per_second_720p")]
    OutputPerSecond720p,
    #[strum(serialize = "output_cost_per_second_768p")]
    OutputPerSecond768p,
    #[strum(serialize = "output_cost_per_token")]
    OutputPerToken,
    #[strum(serialize = "output_cost_per_reasoning_token")]
    OutputReasoningToken,
    #[strum(serialize = "output_cost_per_video_per_second")]
    OutputVideoPerSecond,
    #[strum(serialize = "output_cost_per_video_token")]
    OutputVideoToken,
    #[strum(serialize = "vector_store_cost_per_gb_per_day")]
    VectorStorePerGbPerDay,
}

impl Metric {
    pub fn as_str(self) -> &'static str {
        self.into()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RateKey {
    pub metric: Metric,
    pub threshold: Option<u64>,
    pub tier: Option<ServiceTier>,
    pub batch: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TokenizeError {
    #[error("pricing key names no known metric")]
    UnrecognizedMetric,
    #[error("pricing key threshold is not a token count")]
    BadThreshold,
}

pub fn tokenize(key: &str) -> Result<RateKey, TokenizeError> {
    let (base, batch) = match key.strip_suffix("_batches") {
        Some(base) => (base, true),
        None => (key, false),
    };
    let (base, tier) = ServiceTier::SUFFIXES
        .into_iter()
        .find_map(|tier| {
            base.strip_suffix(format!("_{}", tier.as_str()).as_str())
                .map(|stripped| (stripped, Some(tier)))
        })
        .unwrap_or((base, None));
    let (metric_name, threshold) = match base.strip_suffix("_tokens") {
        Some(prefix) => match prefix.rsplit_once("_above") {
            Some((metric_name, tail)) => (
                metric_name,
                Some(parse_threshold_number(
                    tail.strip_prefix('_').unwrap_or(tail),
                )?),
            ),
            None => (base, None),
        },
        None => (base, None),
    };
    let metric = metric_name
        .parse::<Metric>()
        .map_err(|_| TokenizeError::UnrecognizedMetric)?;
    Ok(RateKey {
        metric,
        threshold,
        tier,
        batch,
    })
}

fn parse_threshold_number(text: &str) -> Result<u64, TokenizeError> {
    let (digits, multiplier) = match text.strip_suffix('k') {
        Some(digits) => (digits, 1_000_u64),
        None => (text, 1),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TokenizeError::BadThreshold);
    }
    digits
        .parse::<u64>()
        .ok()
        .and_then(|value| value.checked_mul(multiplier))
        .ok_or(TokenizeError::BadThreshold)
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdPricing {
    pub above_tokens: u64,
    pub standard: BTreeMap<Metric, Rate>,
    pub tiers: BTreeMap<(Metric, ServiceTier), Rate>,
    pub batches: BTreeMap<Metric, Rate>,
    pub batch_tiers: BTreeMap<(Metric, ServiceTier), Rate>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OffPeakWindow {
    pub hours_utc: Vec<String>,
    pub weekdays: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OffPeakPricing {
    pub input: Rate,
    pub output: Rate,
    pub cache_read: Rate,
    pub cache_creation: Rate,
    pub output_reasoning: Rate,
    pub hours_utc: Vec<String>,
    pub weekdays: Option<Vec<String>>,
    pub weekday_timezone: Option<String>,
    pub windows: Vec<OffPeakWindow>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TieredPricingTier {
    pub range: (f64, Option<f64>),
    pub input: Rate,
    pub output: Rate,
    pub output_reasoning: Rate,
    pub cache_read: Rate,
    pub cache_creation: Rate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SearchContextCostPerQuery {
    pub low: Rate,
    pub medium: Rate,
    pub high: Rate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PtuPricing {
    pub count: i64,
    pub cost_per_hour: Rate,
    pub effective_from: Option<String>,
    pub effective_to: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelPricing {
    pub base: BTreeMap<Metric, Rate>,
    pub tiers: BTreeMap<(Metric, ServiceTier), Rate>,
    pub batches: BTreeMap<Metric, Rate>,
    pub batch_tiers: BTreeMap<(Metric, ServiceTier), Rate>,
    pub thresholds: Vec<ThresholdPricing>,
    pub off_peak: Option<OffPeakPricing>,
    pub tiered_pricing: Option<Vec<TieredPricingTier>>,
    pub search_context_cost_per_query: Option<SearchContextCostPerQuery>,
    pub provider_specific_entry: BTreeMap<String, Rate>,
    pub guardrail_cost_per_unit: BTreeMap<String, Rate>,
    pub web_search_billing_unit: Option<String>,
    pub regional_processing_eu: Rate,
    pub regional_processing_us: Rate,
    pub regional_endpoint_uplift: Rate,
    pub litellm_provider: Option<String>,
    pub ptu: Option<PtuPricing>,
    pub extra: BTreeMap<String, Rate>,
}

impl ModelPricing {
    pub fn rate(&self, metric: Metric) -> Rate {
        self.base.get(&metric).copied().unwrap_or(Rate::Missing)
    }

    pub fn tier_rate(&self, metric: Metric, tier: ServiceTier) -> Rate {
        self.tiers
            .get(&(metric, tier))
            .copied()
            .unwrap_or(Rate::Missing)
    }

    pub fn batch_rate(&self, metric: Metric) -> Rate {
        self.batches.get(&metric).copied().unwrap_or(Rate::Missing)
    }
}
