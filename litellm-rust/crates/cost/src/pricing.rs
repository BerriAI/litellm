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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
        match self {
            Self::Auto => "auto",
            Self::Flex => "flex",
            Self::Priority => "priority",
            Self::Fast => "fast",
            Self::Ultrafast => "ultrafast",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Metric {
    AnnotationPerPage,
    CacheCreationAudioToken,
    CacheCreationToken,
    CacheCreationToken1hr,
    CacheReadAudioToken,
    CacheReadImageToken,
    CacheReadToken,
    CitationToken,
    CodeInterpreterPerSession,
    ComputerUseInput1kTokens,
    ComputerUseOutput1kTokens,
    FileSearchPer1kCalls,
    FileSearchPerGbPerDay,
    GoogleMapsGroundingPerQuery,
    InputAudioPerSecond,
    InputAudioToken,
    InputCharacter,
    InputDbuToken,
    InputPerImage,
    InputPerImageToken,
    InputPerPixel,
    InputPerQuery,
    InputPerRequest,
    InputPerSecond,
    InputPerToken,
    InputPerTokenCacheHit,
    InputVideoPerSecond,
    InputVideoPerSecond15sInterval,
    InputVideoPerSecond8sInterval,
    InputVideoToken,
    OcrPerCredit,
    OcrPerPage,
    OutputAudioToken,
    OutputCharacter,
    OutputDbuToken,
    OutputPerImage,
    OutputPerImage1024,
    OutputPerImage1536,
    OutputPerImage512,
    OutputPerImageToken,
    OutputPerPixel,
    OutputPerSecond,
    OutputPerSecond1080p,
    OutputPerSecond2k,
    OutputPerSecond480p,
    OutputPerSecond4k,
    OutputPerSecond720p,
    OutputPerSecond768p,
    OutputPerToken,
    OutputReasoningToken,
    OutputVideoPerSecond,
    OutputVideoToken,
    VectorStorePerGbPerDay,
}

impl Metric {
    const NAMES: [(Self, &'static str); 53] = [
        (Self::AnnotationPerPage, "annotation_cost_per_page"),
        (
            Self::CacheCreationAudioToken,
            "cache_creation_input_audio_token_cost",
        ),
        (Self::CacheCreationToken, "cache_creation_input_token_cost"),
        (
            Self::CacheCreationToken1hr,
            "cache_creation_input_token_cost_above_1hr",
        ),
        (
            Self::CacheReadAudioToken,
            "cache_read_input_audio_token_cost",
        ),
        (
            Self::CacheReadImageToken,
            "cache_read_input_image_token_cost",
        ),
        (Self::CacheReadToken, "cache_read_input_token_cost"),
        (Self::CitationToken, "citation_cost_per_token"),
        (
            Self::CodeInterpreterPerSession,
            "code_interpreter_cost_per_session",
        ),
        (
            Self::ComputerUseInput1kTokens,
            "computer_use_input_cost_per_1k_tokens",
        ),
        (
            Self::ComputerUseOutput1kTokens,
            "computer_use_output_cost_per_1k_tokens",
        ),
        (Self::FileSearchPer1kCalls, "file_search_cost_per_1k_calls"),
        (
            Self::FileSearchPerGbPerDay,
            "file_search_cost_per_gb_per_day",
        ),
        (
            Self::GoogleMapsGroundingPerQuery,
            "google_maps_grounding_cost_per_query",
        ),
        (Self::InputAudioPerSecond, "input_cost_per_audio_per_second"),
        (Self::InputAudioToken, "input_cost_per_audio_token"),
        (Self::InputCharacter, "input_cost_per_character"),
        (Self::InputDbuToken, "input_dbu_cost_per_token"),
        (Self::InputPerImage, "input_cost_per_image"),
        (Self::InputPerImageToken, "input_cost_per_image_token"),
        (Self::InputPerPixel, "input_cost_per_pixel"),
        (Self::InputPerQuery, "input_cost_per_query"),
        (Self::InputPerRequest, "input_cost_per_request"),
        (Self::InputPerSecond, "input_cost_per_second"),
        (Self::InputPerToken, "input_cost_per_token"),
        (
            Self::InputPerTokenCacheHit,
            "input_cost_per_token_cache_hit",
        ),
        (Self::InputVideoPerSecond, "input_cost_per_video_per_second"),
        (
            Self::InputVideoPerSecond15sInterval,
            "input_cost_per_video_per_second_above_15s_interval",
        ),
        (
            Self::InputVideoPerSecond8sInterval,
            "input_cost_per_video_per_second_above_8s_interval",
        ),
        (Self::InputVideoToken, "input_cost_per_video_token"),
        (Self::OcrPerCredit, "ocr_cost_per_credit"),
        (Self::OcrPerPage, "ocr_cost_per_page"),
        (Self::OutputAudioToken, "output_cost_per_audio_token"),
        (Self::OutputCharacter, "output_cost_per_character"),
        (Self::OutputDbuToken, "output_dbu_cost_per_token"),
        (Self::OutputPerImage, "output_cost_per_image"),
        (Self::OutputPerImage1024, "output_cost_per_image_1024"),
        (Self::OutputPerImage1536, "output_cost_per_image_1536"),
        (Self::OutputPerImage512, "output_cost_per_image_512"),
        (Self::OutputPerImageToken, "output_cost_per_image_token"),
        (Self::OutputPerPixel, "output_cost_per_pixel"),
        (Self::OutputPerSecond, "output_cost_per_second"),
        (Self::OutputPerSecond1080p, "output_cost_per_second_1080p"),
        (Self::OutputPerSecond2k, "output_cost_per_second_2k"),
        (Self::OutputPerSecond480p, "output_cost_per_second_480p"),
        (Self::OutputPerSecond4k, "output_cost_per_second_4k"),
        (Self::OutputPerSecond720p, "output_cost_per_second_720p"),
        (Self::OutputPerSecond768p, "output_cost_per_second_768p"),
        (Self::OutputPerToken, "output_cost_per_token"),
        (
            Self::OutputReasoningToken,
            "output_cost_per_reasoning_token",
        ),
        (
            Self::OutputVideoPerSecond,
            "output_cost_per_video_per_second",
        ),
        (Self::OutputVideoToken, "output_cost_per_video_token"),
        (
            Self::VectorStorePerGbPerDay,
            "vector_store_cost_per_gb_per_day",
        ),
    ];

    pub fn parse(name: &str) -> Option<Self> {
        Self::NAMES
            .iter()
            .find(|(_, known)| *known == name)
            .map(|(metric, _)| *metric)
    }

    pub fn as_str(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(metric, _)| *metric == self)
            .map(|(_, name)| *name)
            .expect("every metric carries its name")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RateKey {
    pub metric: Metric,
    pub threshold: Option<u64>,
    pub tier: Option<ServiceTier>,
    pub batch: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenizeError {
    UnrecognizedMetric,
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
    let metric = Metric::parse(metric_name).ok_or(TokenizeError::UnrecognizedMetric)?;
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
