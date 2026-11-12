use std::num::NonZeroU64;

use litellm_cost::price::{
    Band, CacheReadMode, CacheTtl, Channel, Feature, ImageSize, InputUnit, Meter, OutputUnit,
    Resolution, ServiceTier, TokenKind, When,
};

/// A flat price key as the grammar reads it: `{meter}[_above_{N}k_tokens][_{tier}]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatKey {
    pub meter: Meter,
    pub band: Band,
    pub tier: Option<TierSuffix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierSuffix {
    Service(ServiceTier),
    Batches,
    Decisions,
}

impl FlatKey {
    pub fn parse(key: &str) -> Option<Self> {
        let (rest, tier) = split_tier(key);
        let (base, suffix) = split_above(rest)?;
        let (meter, band) = match suffix {
            Above::None => (meter(base)?, Band::Base),
            Above::PromptTokens(tokens) => (meter(base)?, Band::over_prompt_tokens(tokens)),
            Above::Seconds(seconds) => match meter(base)? {
                Meter::Input(InputUnit::VideoSecond { over_seconds: None }) => (
                    Meter::Input(InputUnit::VideoSecond {
                        over_seconds: NonZeroU64::new(seconds),
                    }),
                    Band::Base,
                ),
                _ => return None,
            },
        };
        Some(Self { meter, band, tier })
    }

    pub fn has_suffix(&self) -> bool {
        self.band != Band::Base || self.tier.is_some()
    }

    pub fn channel_and_when(&self) -> (Channel, When) {
        let (channel, tier) = match self.tier {
            None => (Channel::Online, ServiceTier::Standard),
            Some(TierSuffix::Service(tier)) => (Channel::Online, tier),
            Some(TierSuffix::Batches) => (Channel::Batch, ServiceTier::Standard),
            Some(TierSuffix::Decisions) => (Channel::Decisions, ServiceTier::Standard),
        };
        (
            channel,
            When {
                tier,
                band: self.band,
            },
        )
    }
}

enum Above {
    None,
    PromptTokens(u64),
    Seconds(u64),
}

fn split_tier(key: &str) -> (&str, Option<TierSuffix>) {
    let Some((head, tail)) = key.rsplit_once('_') else {
        return (key, None);
    };
    let tier = match tail {
        "priority" => TierSuffix::Service(ServiceTier::Priority),
        "flex" => TierSuffix::Service(ServiceTier::Flex),
        "ultrafast" => TierSuffix::Service(ServiceTier::Ultrafast),
        "balanced" => TierSuffix::Service(ServiceTier::Balanced),
        "batches" => TierSuffix::Batches,
        "decisions" => TierSuffix::Decisions,
        _ => return (key, None),
    };
    (head, Some(tier))
}

fn split_above(key: &str) -> Option<(&str, Above)> {
    let Some((head, tail)) = key.rsplit_once("_above_") else {
        return Some((key, Above::None));
    };
    if tail == "1hr" || tail == "12hr" {
        return Some((key, Above::None));
    }
    if let Some(thousands) = tail.strip_suffix("k_tokens") {
        return Some((
            head,
            Above::PromptTokens(number(thousands)?.checked_mul(1000)?),
        ));
    }
    if let Some(tokens) = tail.strip_suffix("_tokens") {
        return Some((head, Above::PromptTokens(number(tokens)?)));
    }
    let seconds = tail.strip_suffix("s_interval")?;
    Some((head, Above::Seconds(number(seconds)?)))
}

fn number(digits: &str) -> Option<u64> {
    (!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn meter(base: &str) -> Option<Meter> {
    let fixed = match base {
        "cost_per_second" => Some(Meter::WallClockSecond),
        "explicit_cache_read_input_token_cost" => Some(Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Explicit,
        }),
        "annotation_cost_per_page" => Some(Meter::Feature(Feature::AnnotationPage)),
        "citation_cost_per_token" => Some(Meter::Feature(Feature::CitationToken)),
        "code_interpreter_cost_per_session" => {
            Some(Meter::Feature(Feature::CodeInterpreterSession))
        }
        "google_maps_grounding_cost_per_query" => {
            Some(Meter::Feature(Feature::GoogleMapsGroundingQuery))
        }
        "ocr_cost_per_credit" => Some(Meter::Feature(Feature::OcrCredit)),
        "ocr_cost_per_page" => Some(Meter::Feature(Feature::OcrPage)),
        "browser_open_cost_per_request" => Some(Meter::Feature(Feature::BrowserOpen)),
        "cost_per_additional_result" => Some(Meter::Feature(Feature::AdditionalResult)),
        _ => None,
    };
    if fixed.is_some() {
        return fixed;
    }
    if let Some(unit) = base.strip_prefix("input_cost_per_") {
        return input_unit(unit).map(Meter::Input);
    }
    if let Some(unit) = base.strip_prefix("output_cost_per_") {
        return output_unit(unit).map(Meter::Output);
    }
    if let Some(kind) = base
        .strip_prefix("cache_read_input_")
        .and_then(|rest| rest.strip_suffix("_cost"))
    {
        return Some(Meter::CacheRead {
            kind: token_kind(kind)?,
            mode: CacheReadMode::Implicit,
        });
    }
    let rest = base.strip_prefix("cache_creation_input_")?;
    let (kind, ttl) = if let Some(kind) = rest.strip_suffix("_cost_above_1hr") {
        (kind, CacheTtl::OneHour)
    } else if let Some(kind) = rest.strip_suffix("_cost_above_12hr") {
        (kind, CacheTtl::TwelveHours)
    } else {
        (rest.strip_suffix("_cost")?, CacheTtl::Default)
    };
    Some(Meter::CacheWrite {
        kind: token_kind(kind)?,
        ttl,
    })
}

fn token_kind(unit: &str) -> Option<TokenKind> {
    match unit {
        "token" => Some(TokenKind::Text),
        "audio_token" => Some(TokenKind::Audio),
        "image_token" => Some(TokenKind::Image),
        "video_token" => Some(TokenKind::Video),
        _ => None,
    }
}

fn input_unit(unit: &str) -> Option<InputUnit> {
    if let Some(kind) = token_kind(unit) {
        return Some(InputUnit::Token(kind));
    }
    match unit {
        "character" => Some(InputUnit::Character),
        "image" => Some(InputUnit::Image),
        "pixel" => Some(InputUnit::Pixel),
        "query" => Some(InputUnit::Query { over_results: None }),
        "request" => Some(InputUnit::Request),
        "second" => Some(InputUnit::Second),
        "text_message" => Some(InputUnit::TextMessage),
        "audio_per_second" => Some(InputUnit::AudioSecond),
        "video_per_second" => Some(InputUnit::VideoSecond { over_seconds: None }),
        _ => None,
    }
}

fn output_unit(unit: &str) -> Option<OutputUnit> {
    if let Some(kind) = token_kind(unit) {
        return Some(OutputUnit::Token(kind));
    }
    match unit.split_once('_') {
        Some(("image", label)) => {
            return ImageSize::new(label).map(|size| OutputUnit::Image(Some(size)));
        }
        Some(("second", label)) => {
            return Resolution::new(label).map(|resolution| OutputUnit::Second(Some(resolution)));
        }
        _ => {}
    }
    match unit {
        "reasoning_token" => Some(OutputUnit::ReasoningToken),
        "character" => Some(OutputUnit::Character),
        "image" => Some(OutputUnit::Image(None)),
        "pixel" => Some(OutputUnit::Pixel),
        "second" => Some(OutputUnit::Second(None)),
        "video_per_second" => Some(OutputUnit::VideoSecond),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const TEXT_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Text));
    const TEXT_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Text));

    fn key(meter: Meter, band: Band, tier: Option<TierSuffix>) -> FlatKey {
        FlatKey { meter, band, tier }
    }

    #[rstest]
    #[case::input_token("input_cost_per_token", key(TEXT_INPUT, Band::Base, None))]
    #[case::band_and_tier(
        "output_cost_per_token_above_272k_tokens_priority",
        key(
            TEXT_OUTPUT,
            Band::over_prompt_tokens(272_000),
            Some(TierSuffix::Service(ServiceTier::Priority))
        )
    )]
    #[case::band_without_k(
        "input_cost_per_token_above_128_tokens",
        key(TEXT_INPUT, Band::over_prompt_tokens(128), None)
    )]
    #[case::zero_band_is_base(
        "input_cost_per_token_above_0k_tokens",
        key(TEXT_INPUT, Band::Base, None)
    )]
    #[case::batch(
        "input_cost_per_token_batches",
        key(TEXT_INPUT, Band::Base, Some(TierSuffix::Batches))
    )]
    #[case::reasoning(
        "output_cost_per_reasoning_token",
        key(Meter::Output(OutputUnit::ReasoningToken), Band::Base, None)
    )]
    #[case::cache_read_audio(
        "cache_read_input_audio_token_cost",
        key(Meter::CacheRead { kind: TokenKind::Audio, mode: CacheReadMode::Implicit }, Band::Base, None)
    )]
    #[case::explicit_cache_read(
        "explicit_cache_read_input_token_cost",
        key(Meter::CacheRead { kind: TokenKind::Text, mode: CacheReadMode::Explicit }, Band::Base, None)
    )]
    #[case::cache_write_one_hour_band(
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens",
        key(
            Meter::CacheWrite { kind: TokenKind::Text, ttl: CacheTtl::OneHour },
            Band::over_prompt_tokens(200_000),
            None
        )
    )]
    #[case::cache_write_twelve_hours_priority(
        "cache_creation_input_token_cost_above_12hr_above_272k_tokens_priority",
        key(
            Meter::CacheWrite { kind: TokenKind::Text, ttl: CacheTtl::TwelveHours },
            Band::over_prompt_tokens(272_000),
            Some(TierSuffix::Service(ServiceTier::Priority))
        )
    )]
    #[case::image_label(
        "output_cost_per_image_0.5K",
        key(
            Meter::Output(OutputUnit::Image(ImageSize::new("0.5K"))),
            Band::Base,
            None
        )
    )]
    #[case::image_token_is_a_token(
        "output_cost_per_image_token",
        key(Meter::Output(OutputUnit::Token(TokenKind::Image)), Band::Base, None)
    )]
    #[case::resolution(
        "output_cost_per_second_720p",
        key(
            Meter::Output(OutputUnit::Second(Resolution::new("720p"))),
            Band::Base,
            None
        )
    )]
    #[case::video_interval(
        "input_cost_per_video_per_second_above_8s_interval",
        key(Meter::Input(InputUnit::VideoSecond { over_seconds: NonZeroU64::new(8) }), Band::Base, None)
    )]
    #[case::feature_batch(
        "ocr_cost_per_page_batches",
        key(
            Meter::Feature(Feature::OcrPage),
            Band::Base,
            Some(TierSuffix::Batches)
        )
    )]
    #[case::wall_clock("cost_per_second", key(Meter::WallClockSecond, Band::Base, None))]
    #[case::browser_open(
        "browser_open_cost_per_request",
        key(Meter::Feature(Feature::BrowserOpen), Band::Base, None)
    )]
    #[case::additional_result(
        "cost_per_additional_result",
        key(Meter::Feature(Feature::AdditionalResult), Band::Base, None)
    )]
    fn parses_every_key_shape(#[case] text: &str, #[case] expected: FlatKey) {
        assert_eq!(FlatKey::parse(text), Some(expected));
    }

    #[rstest]
    #[case::typo("input_cost_per_tokne")]
    #[case::unused_meter("cost_per_token")]
    #[case::other_currency("input_dbu_cost_per_token")]
    #[case::cache_hit_alias("input_cost_per_token_cache_hit")]
    #[case::non_numeric_band("input_cost_per_token_above_xk_tokens")]
    #[case::interval_on_a_token_meter("input_cost_per_token_above_8s_interval")]
    #[case::read_has_no_ttl("cache_read_input_token_cost_above_1hr")]
    #[case::cache_unit_must_be_token("cache_read_input_character_cost")]
    #[case::bad_image_size("output_cost_per_image_big")]
    #[case::bad_resolution("output_cost_per_second_hd")]
    #[case::unknown_feature("telepathy_cost_per_page")]
    #[case::empty("")]
    fn rejects_keys_outside_the_grammar(#[case] text: &str) {
        assert_eq!(FlatKey::parse(text), None);
    }

    #[rstest]
    #[case::plain("input_cost_per_token", (Channel::Online, When::default()))]
    #[case::service_tier(
        "input_cost_per_token_flex",
        (Channel::Online, When { tier: ServiceTier::Flex, band: Band::Base })
    )]
    #[case::batch(
        "input_cost_per_token_above_200k_tokens_batches",
        (Channel::Batch, When { tier: ServiceTier::Standard, band: Band::over_prompt_tokens(200_000) })
    )]
    #[case::decisions("input_cost_per_token_decisions", (Channel::Decisions, When::default()))]
    fn batch_and_decisions_pick_a_channel_not_a_tier(
        #[case] text: &str,
        #[case] expected: (Channel, When),
    ) {
        assert_eq!(FlatKey::parse(text).unwrap().channel_and_when(), expected);
    }
}
