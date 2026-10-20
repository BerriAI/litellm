use std::fmt;
use std::str::FromStr;

use crate::error::InvalidRateKey;

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde_with::SerializeDisplay,
    serde_with::DeserializeFromStr,
)]
pub struct Rate {
    pub charge: Charge,
    pub above: Option<Threshold>,
    pub tier: Option<Tier>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Charge {
    Input(Unit),
    Output(Unit),
    CacheRead(TokenKind),
    CacheWrite(TokenKind, CacheTtl),
    Feature(Feature, Unit),
    Usage(Unit),
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum Feature {
    #[strum(serialize = "annotation")]
    Annotation,
    #[strum(serialize = "ocr")]
    Ocr,
    #[strum(serialize = "citation")]
    Citation,
    #[strum(serialize = "code_interpreter")]
    CodeInterpreter,
    #[strum(serialize = "google_maps_grounding")]
    GoogleMapsGrounding,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Unit {
    Token(TokenKind),
    Simple(SimpleUnit),
    Image(Option<ImageSize>),
    Second(Option<Resolution>),
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum TokenKind {
    #[strum(serialize = "token")]
    Text,
    #[strum(serialize = "audio_token")]
    Audio,
    #[strum(serialize = "image_token")]
    Image,
    #[strum(serialize = "video_token")]
    Video,
    #[strum(serialize = "reasoning_token")]
    Reasoning,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum SimpleUnit {
    #[strum(serialize = "character")]
    Character,
    #[strum(serialize = "pixel")]
    Pixel,
    #[strum(serialize = "query")]
    Query,
    #[strum(serialize = "request")]
    Request,
    #[strum(serialize = "text_message")]
    TextMessage,
    #[strum(serialize = "audio_per_second")]
    AudioSecond,
    #[strum(serialize = "video_per_second")]
    VideoSecond,
    #[strum(serialize = "page")]
    Page,
    #[strum(serialize = "credit")]
    Credit,
    #[strum(serialize = "session")]
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheTtl {
    Default,
    OneHour,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Threshold {
    KiloTokens(u32),
    Seconds(u32),
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    strum::EnumString,
    strum::IntoStaticStr,
    strum::VariantArray,
)]
pub enum Tier {
    #[strum(serialize = "batches")]
    Batch,
    #[strum(serialize = "priority")]
    Priority,
    #[strum(serialize = "flex")]
    Flex,
    #[strum(serialize = "decisions")]
    Decisions,
    #[strum(serialize = "ultrafast")]
    Ultrafast,
    #[strum(serialize = "balanced")]
    Balanced,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageSize(Box<str>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Resolution(Box<str>);

const ONE_HOUR_TTL: &str = "_above_1hr";

impl ImageSize {
    pub fn new(label: &str) -> Option<Self> {
        let number = label.strip_suffix('K').unwrap_or(label);
        let (whole, fraction) = number.split_once('.').unwrap_or((number, "0"));
        (is_digits(whole) && is_digits(fraction)).then(|| Self(label.into()))
    }
}

impl Resolution {
    pub fn new(label: &str) -> Option<Self> {
        let digits = label.strip_suffix(['p', 'k'])?;
        is_digits(digits).then(|| Self(label.into()))
    }
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

impl FromStr for Rate {
    type Err = InvalidRateKey;

    fn from_str(key: &str) -> Result<Self, Self::Err> {
        let (rest, tier) = split_tier(key);
        let (rest, above) = split_threshold(rest)?;
        Ok(Self {
            charge: rest.parse()?,
            above,
            tier,
        })
    }
}

fn split_tier(key: &str) -> (&str, Option<Tier>) {
    key.rsplit_once('_')
        .and_then(|(head, tail)| Some((head, tail.parse().ok()?)))
        .map_or((key, None), |(head, tier)| (head, Some(tier)))
}

fn split_threshold(key: &str) -> Result<(&str, Option<Threshold>), InvalidRateKey> {
    let Some((head, tail)) = key.rsplit_once("_above_") else {
        return Ok((key, None));
    };
    if tail == "1hr" {
        return Ok((key, None));
    }
    let threshold = match (
        tail.strip_suffix("k_tokens"),
        tail.strip_suffix("s_interval"),
    ) {
        (Some(count), _) => Threshold::KiloTokens(count.parse().map_err(|_| InvalidRateKey)?),
        (_, Some(count)) => Threshold::Seconds(count.parse().map_err(|_| InvalidRateKey)?),
        (None, None) => return Err(InvalidRateKey),
    };
    Ok((head, Some(threshold)))
}

impl FromStr for Charge {
    type Err = InvalidRateKey;

    fn from_str(key: &str) -> Result<Self, Self::Err> {
        if let Some(unit) = key.strip_prefix("cost_per_") {
            return Ok(Self::Usage(unit.parse()?));
        }
        if let Some((payer, unit)) = key.split_once("_cost_per_") {
            return match payer {
                "input" => Ok(Self::Input(unit.parse()?)),
                "output" => Ok(Self::Output(unit.parse()?)),
                feature => Ok(Self::Feature(
                    feature.parse().map_err(|_| InvalidRateKey)?,
                    unit.parse()?,
                )),
            };
        }
        if let Some(kind) = key
            .strip_prefix("cache_read_input_")
            .and_then(|rest| rest.strip_suffix("_cost"))
        {
            return Ok(Self::CacheRead(kind.parse().map_err(|_| InvalidRateKey)?));
        }
        let rest = key
            .strip_prefix("cache_creation_input_")
            .ok_or(InvalidRateKey)?;
        let (rest, ttl) = rest
            .strip_suffix(ONE_HOUR_TTL)
            .map_or((rest, CacheTtl::Default), |rest| (rest, CacheTtl::OneHour));
        let kind = rest.strip_suffix("_cost").ok_or(InvalidRateKey)?;
        Ok(Self::CacheWrite(
            kind.parse().map_err(|_| InvalidRateKey)?,
            ttl,
        ))
    }
}

impl FromStr for Unit {
    type Err = InvalidRateKey;

    fn from_str(unit: &str) -> Result<Self, Self::Err> {
        if let Ok(kind) = unit.parse() {
            return Ok(Self::Token(kind));
        }
        if let Ok(simple) = unit.parse() {
            return Ok(Self::Simple(simple));
        }
        match unit.split_once('_') {
            None if unit == "image" => Ok(Self::Image(None)),
            None if unit == "second" => Ok(Self::Second(None)),
            Some(("image", label)) => ImageSize::new(label)
                .map(|size| Self::Image(Some(size)))
                .ok_or(InvalidRateKey),
            Some(("second", label)) => Resolution::new(label)
                .map(|resolution| Self::Second(Some(resolution)))
                .ok_or(InvalidRateKey),
            _ => Err(InvalidRateKey),
        }
    }
}

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.charge)?;
        match self.above {
            Some(Threshold::KiloTokens(count)) => write!(f, "_above_{count}k_tokens")?,
            Some(Threshold::Seconds(count)) => write!(f, "_above_{count}s_interval")?,
            None => {}
        }
        match self.tier {
            Some(tier) => write!(f, "_{}", <&str>::from(tier)),
            None => Ok(()),
        }
    }
}

impl fmt::Display for Charge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(unit) => write!(f, "input_cost_per_{unit}"),
            Self::Output(unit) => write!(f, "output_cost_per_{unit}"),
            Self::CacheRead(kind) => write!(f, "cache_read_input_{}_cost", <&str>::from(*kind)),
            Self::CacheWrite(kind, ttl) => {
                let ttl = match ttl {
                    CacheTtl::Default => "",
                    CacheTtl::OneHour => ONE_HOUR_TTL,
                };
                write!(f, "cache_creation_input_{}_cost{ttl}", <&str>::from(*kind))
            }
            Self::Feature(feature, unit) => {
                write!(f, "{}_cost_per_{unit}", <&str>::from(*feature))
            }
            Self::Usage(unit) => write!(f, "cost_per_{unit}"),
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Token(kind) => f.write_str(kind.into()),
            Self::Simple(simple) => f.write_str(simple.into()),
            Self::Image(None) => f.write_str("image"),
            Self::Image(Some(ImageSize(label))) => write!(f, "image_{label}"),
            Self::Second(None) => f.write_str("second"),
            Self::Second(Some(Resolution(label))) => write!(f, "second_{label}"),
        }
    }
}

#[cfg(feature = "schema")]
pub(crate) fn key_pattern() -> String {
    use strum::VariantArray;

    fn alternatives<T: Copy + Into<&'static str>>(variants: &[T]) -> String {
        variants
            .iter()
            .map(|variant| (*variant).into())
            .collect::<Vec<_>>()
            .join("|")
    }
    let tokens = alternatives(TokenKind::VARIANTS);
    let simple = alternatives(SimpleUnit::VARIANTS);
    let tiers = alternatives(Tier::VARIANTS);
    let features = alternatives(Feature::VARIANTS);
    let unit = format!(r"{tokens}|{simple}|image(_\d+(\.\d+)?K?)?|second(_\d+[pk])?");
    format!(
        r"^((input|output|{features})_cost_per_({unit})|cost_per_({unit})|cache_read_input_({tokens})_cost|cache_creation_input_({tokens})_cost({ONE_HOUR_TTL})?)(_above_(\d+k_tokens|\d+s_interval))?(_({tiers}))?$"
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn rate(charge: Charge, above: Option<Threshold>, tier: Option<Tier>) -> Rate {
        Rate {
            charge,
            above,
            tier,
        }
    }

    #[rstest]
    #[case::input_token(
        "input_cost_per_token",
        rate(Charge::Input(Unit::Token(TokenKind::Text)), None, None)
    )]
    #[case::output_threshold_tier(
        "output_cost_per_token_above_272k_tokens_priority",
        rate(
            Charge::Output(Unit::Token(TokenKind::Text)),
            Some(Threshold::KiloTokens(272)),
            Some(Tier::Priority)
        )
    )]
    #[case::reasoning(
        "output_cost_per_reasoning_token",
        rate(Charge::Output(Unit::Token(TokenKind::Reasoning)), None, None)
    )]
    #[case::cache_read_tier(
        "cache_read_input_token_cost_decisions",
        rate(Charge::CacheRead(TokenKind::Text), None, Some(Tier::Decisions))
    )]
    #[case::cache_read_audio(
        "cache_read_input_audio_token_cost",
        rate(Charge::CacheRead(TokenKind::Audio), None, None)
    )]
    #[case::cache_write(
        "cache_creation_input_token_cost_batches",
        rate(
            Charge::CacheWrite(TokenKind::Text, CacheTtl::Default),
            None,
            Some(Tier::Batch)
        )
    )]
    #[case::cache_write_one_hour(
        "cache_creation_input_token_cost_above_1hr",
        rate(Charge::CacheWrite(TokenKind::Text, CacheTtl::OneHour), None, None)
    )]
    #[case::cache_write_one_hour_threshold(
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens",
        rate(
            Charge::CacheWrite(TokenKind::Text, CacheTtl::OneHour),
            Some(Threshold::KiloTokens(200)),
            None
        )
    )]
    #[case::image_label(
        "output_cost_per_image_0.5K",
        rate(Charge::Output(Unit::Image(ImageSize::new("0.5K"))), None, None)
    )]
    #[case::image_pixels(
        "output_cost_per_image_1024",
        rate(Charge::Output(Unit::Image(ImageSize::new("1024"))), None, None)
    )]
    #[case::image(
        "input_cost_per_image",
        rate(Charge::Input(Unit::Image(None)), None, None)
    )]
    #[case::resolution(
        "output_cost_per_second_720p",
        rate(Charge::Output(Unit::Second(Resolution::new("720p"))), None, None)
    )]
    #[case::second(
        "input_cost_per_second",
        rate(Charge::Input(Unit::Second(None)), None, None)
    )]
    #[case::video_interval(
        "input_cost_per_video_per_second_above_8s_interval",
        rate(
            Charge::Input(Unit::Simple(SimpleUnit::VideoSecond)),
            Some(Threshold::Seconds(8)),
            None
        )
    )]
    #[case::text_message(
        "input_cost_per_text_message",
        rate(Charge::Input(Unit::Simple(SimpleUnit::TextMessage)), None, None)
    )]
    #[case::feature_tier(
        "ocr_cost_per_page_batches",
        rate(
            Charge::Feature(Feature::Ocr, Unit::Simple(SimpleUnit::Page)),
            None,
            Some(Tier::Batch)
        )
    )]
    #[case::feature_token(
        "citation_cost_per_token",
        rate(
            Charge::Feature(Feature::Citation, Unit::Token(TokenKind::Text)),
            None,
            None
        )
    )]
    #[case::usage("cost_per_second", rate(Charge::Usage(Unit::Second(None)), None, None))]
    fn parses_and_displays_the_same_key(#[case] key: &str, #[case] expected: Rate) {
        let parsed: Rate = key.parse().unwrap();

        assert_eq!(parsed, expected);
        assert_eq!(parsed.to_string(), key);
    }

    #[rstest]
    #[case::typo("input_cost_per_tokne")]
    #[case::unknown_feature("dbu_cost_per_token")]
    #[case::other_currency("input_dbu_cost_per_token")]
    #[case::unknown_tier_word("input_cost_per_token_cache_hit")]
    #[case::bare_threshold("input_cost_per_token_above_200_tokens")]
    #[case::non_numeric_threshold("input_cost_per_token_above_xk_tokens")]
    #[case::read_has_no_ttl("cache_read_input_token_cost_above_1hr")]
    #[case::cache_unit_must_be_token("cache_read_input_character_cost")]
    #[case::bad_image_size("output_cost_per_image_big")]
    #[case::bad_resolution("output_cost_per_second_hd")]
    #[case::empty("")]
    fn rejects_keys_outside_the_grammar(#[case] key: &str) {
        assert!(key.parse::<Rate>().is_err());
    }

    #[rstest]
    #[case::input_token("input_cost_per_token", true)]
    #[case::cache_write_one_hour_threshold(
        "cache_creation_input_token_cost_above_1hr_above_200k_tokens",
        true
    )]
    #[case::image_label("output_cost_per_image_0.5K", true)]
    #[case::resolution_tier("output_cost_per_second_4k_flex", true)]
    #[case::feature_tier("annotation_cost_per_page_batches", true)]
    #[case::usage("cost_per_second", true)]
    #[case::other_currency("input_dbu_cost_per_token", false)]
    #[case::typo("input_cost_per_tokne", false)]
    #[case::read_has_no_ttl("cache_read_input_token_cost_above_1hr", false)]
    #[case::bare_threshold("input_cost_per_token_above_200_tokens", false)]
    #[cfg(feature = "schema")]
    fn key_pattern_agrees_with_the_parser(#[case] key: &str, #[case] accepted: bool) {
        let schema = serde_json::json!({
            "type": "object",
            "patternProperties": {key_pattern(): {}},
            "additionalProperties": false,
        });
        let validator = jsonschema::validator_for(&schema).unwrap();

        assert_eq!(key.parse::<Rate>().is_ok(), accepted);
        assert_eq!(validator.is_valid(&serde_json::json!({key: 1})), accepted);
    }
}
