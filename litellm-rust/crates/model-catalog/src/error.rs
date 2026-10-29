use std::num::ParseIntError;

use strum::{EnumDiscriminants, IntoStaticStr, VariantArray};
use thiserror::Error;

#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(Box<Context>);

#[derive(Debug, Error)]
enum Context {
    #[error(transparent)]
    Bare(Repr),
    #[error("{}", display_path(path))]
    At {
        path: String,
        #[source]
        cause: Repr,
    },
}

#[derive(Debug, Error)]
pub(crate) enum DateCause {
    #[error("invalid date component")]
    Number(#[from] ParseIntError),
    #[error("calendar component out of range")]
    Range(#[from] time::error::ComponentRange),
}

#[derive(Debug, Error, EnumDiscriminants)]
#[strum_discriminants(name(ErrorKind), derive(IntoStaticStr, VariantArray), vis(pub))]
pub(crate) enum Repr {
    #[error("invalid JSON")]
    #[strum_discriminants(strum(serialize = "json"))]
    Json(#[source] serde_json::Error),
    #[error("has no models")]
    #[strum_discriminants(strum(serialize = "empty"))]
    Empty,
    #[error("has {actual} models, below minimum {minimum}")]
    #[strum_discriminants(strum(serialize = "below_minimum"))]
    BelowMinimum { actual: usize, minimum: usize },
    #[error("has {actual} models, below {ratio} of reference count {reference}")]
    #[strum_discriminants(strum(serialize = "shrunk"))]
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
    #[error("must be a JSON object")]
    #[strum_discriminants(strum(serialize = "not_an_object"))]
    NotAnObject,
    #[error("invalid value")]
    #[strum_discriminants(strum(serialize = "invalid"))]
    Invalid(#[source] serde_json::Error),
    #[error("not a field this block reads")]
    #[strum_discriminants(strum(serialize = "unknown_block_key"))]
    UnknownBlockKey,
    #[error("required")]
    #[strum_discriminants(strum(serialize = "missing_provider"))]
    MissingProvider,
    #[error("{zone:?} is not an IANA time zone")]
    #[strum_discriminants(strum(serialize = "unknown_timezone"))]
    UnknownTimezone { zone: String },
    #[error("alias {alias:?} is already a model name")]
    #[strum_discriminants(strum(serialize = "alias_shadows_model"))]
    AliasShadowsModel { alias: String },
    #[error("alias {alias:?} already belongs to {taken_by}")]
    #[strum_discriminants(strum(serialize = "alias_taken"))]
    AliasTaken { alias: String, taken_by: String },
    #[error("{first} and {second} price the same meter under the same conditions")]
    #[strum_discriminants(strum(serialize = "duplicate_price"))]
    DuplicatePrice { first: String, second: String },
    #[error("a price inside {block} takes no band or tier suffix")]
    #[strum_discriminants(strum(serialize = "suffix_inside_block"))]
    SuffixInsideBlock { block: &'static str },
    #[error("a tier needs exactly one of range and max_results_range")]
    #[strum_discriminants(strum(serialize = "tier_range"))]
    TierRange,
    #[error("the range ends below its start")]
    #[strum_discriminants(strum(serialize = "tier_reversed"))]
    TierReversed,
    #[error("starts at {start}, but the tiers before it end at {expected}")]
    #[strum_discriminants(strum(serialize = "tier_gap"))]
    TierGap { start: u64, expected: u64 },
    #[error("tiered_pricing mixes token ranges and result ranges")]
    #[strum_discriminants(strum(serialize = "mixed_tier_basis"))]
    MixedTierBasis,
    #[error("a tier by result count only prices input_cost_per_query")]
    #[strum_discriminants(strum(serialize = "not_priced_by_results"))]
    NotPricedByResults,
    #[error("tiered_pricing and prompt token bands are set in two places")]
    #[strum_discriminants(strum(serialize = "bands_from_two_sources"))]
    BandsFromTwoSources,
    #[error("has no off-peak rate")]
    #[strum_discriminants(strum(serialize = "not_an_off_peak_rate"))]
    NotAnOffPeakRate,
    #[error("off_peak_pricing has no prices")]
    #[strum_discriminants(strum(serialize = "no_off_peak_prices"))]
    NoOffPeakPrices,
    #[error("off_peak_pricing needs hours_utc or windows")]
    #[strum_discriminants(strum(serialize = "no_off_peak_hours"))]
    NoOffPeakHours,
    #[error("must list at least one entry")]
    #[strum_discriminants(strum(serialize = "empty_list"))]
    EmptyList,
    #[error("expected a calendar date YYYY-MM-DD")]
    #[strum_discriminants(strum(serialize = "invalid_date"))]
    InvalidDate {
        #[source]
        cause: Option<DateCause>,
    },
    #[error("not a field litellm reads")]
    #[strum_discriminants(strum(serialize = "unread_field"))]
    UnreadField,
    #[error("has no rules")]
    #[strum_discriminants(strum(serialize = "no_fallback_rules"))]
    NoFallbackRules,
}

/// What a problem takes out of the served map
#[derive(Clone, Copy, Debug, PartialEq, Eq, VariantArray)]
pub enum Rejects {
    Document,
    Row,
    Nothing,
}

impl Error {
    pub fn kind(&self) -> ErrorKind {
        ErrorKind::from(self.repr())
    }

    pub fn rejects(&self) -> Rejects {
        match self.repr() {
            Repr::Json(_) | Repr::Empty | Repr::BelowMinimum { .. } | Repr::Shrunk { .. } => {
                Rejects::Document
            }
            Repr::NotAnObject
            | Repr::Invalid(_)
            | Repr::UnknownBlockKey
            | Repr::UnknownTimezone { .. }
            | Repr::AliasShadowsModel { .. }
            | Repr::AliasTaken { .. }
            | Repr::DuplicatePrice { .. }
            | Repr::SuffixInsideBlock { .. }
            | Repr::TierRange
            | Repr::TierReversed
            | Repr::TierGap { .. }
            | Repr::MixedTierBasis
            | Repr::NotPricedByResults
            | Repr::BandsFromTwoSources
            | Repr::NotAnOffPeakRate
            | Repr::NoOffPeakPrices
            | Repr::NoOffPeakHours
            | Repr::EmptyList
            | Repr::InvalidDate { .. } => Rejects::Row,
            Repr::MissingProvider | Repr::UnreadField | Repr::NoFallbackRules => Rejects::Nothing,
        }
    }
}

#[derive(Debug)]
pub struct Loaded<T> {
    pub value: T,
    pub errors: Vec<Error>,
}

impl From<Repr> for Error {
    fn from(cause: Repr) -> Self {
        Self(Box::new(Context::Bare(cause)))
    }
}

impl From<DateCause> for Error {
    fn from(cause: DateCause) -> Self {
        Repr::InvalidDate { cause: Some(cause) }.into()
    }
}

impl Error {
    fn repr(&self) -> &Repr {
        match &*self.0 {
            Context::Bare(cause) | Context::At { cause, .. } => cause,
        }
    }

    pub(crate) fn new(path: impl Into<String>, cause: Repr) -> Self {
        Self(Box::new(Context::At {
            path: path.into(),
            cause,
        }))
    }

    pub(crate) fn from_key(key: &str, cause: Repr) -> Self {
        Self::new(pointer_for_key(key), cause)
    }

    pub(crate) fn from_path(path: &str, cause: Repr) -> Self {
        Self::new(path_to_pointer(path), cause)
    }

    pub fn path(&self) -> &str {
        match &*self.0 {
            Context::Bare(_) => "",
            Context::At { path, .. } => path,
        }
    }

    pub fn key(&self) -> Option<String> {
        pointer_root(self.path())
    }

    pub fn field(&self) -> Option<String> {
        let segments: Vec<_> = self
            .path()
            .split('/')
            .skip(2)
            .map(unescape_pointer_segment)
            .collect();
        (!segments.is_empty()).then(|| segments.join("/"))
    }

    pub(crate) fn prepend(self, key: &str) -> Self {
        let (path, cause) = match *self.0 {
            Context::Bare(cause) => (String::new(), cause),
            Context::At { path, cause } => (path, cause),
        };
        Self::new(format!("/{}{path}", escape_pointer_segment(key)), cause)
    }
}

pub(crate) fn pointer_root(path: &str) -> Option<String> {
    path.strip_prefix('/')
        .and_then(|path| path.split('/').next())
        .map(unescape_pointer_segment)
}

fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

pub(crate) fn pointer_for_key(key: &str) -> String {
    format!("/{}", escape_pointer_segment(key))
}

pub(crate) fn pointer_child(parent: &str, key: &str) -> String {
    format!("{parent}/{}", escape_pointer_segment(key))
}

fn unescape_pointer_segment(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

fn path_to_pointer(key: &str) -> String {
    let mut pointer = String::new();
    let mut segment = String::new();
    let mut chars = key.chars();
    let mut push_segment = |segment: &mut String| {
        if !segment.is_empty() {
            pointer.push('/');
            pointer.push_str(&escape_pointer_segment(segment));
            segment.clear();
        }
    };
    while let Some(character) = chars.next() {
        match character {
            '.' => push_segment(&mut segment),
            '[' => {
                push_segment(&mut segment);
                let mut index = String::new();
                for character in chars.by_ref() {
                    if character == ']' {
                        break;
                    }
                    index.push(character);
                }
                push_segment(&mut index);
            }
            character => segment.push(character),
        }
    }
    push_segment(&mut segment);
    pointer
}

fn display_path(path: &str) -> String {
    if path.is_empty() {
        return "catalog".to_owned();
    }
    path.split('/')
        .skip(1)
        .map(unescape_pointer_segment)
        .enumerate()
        .fold(String::new(), |mut display, (index, segment)| {
            if index == 0 {
                display.push_str(&segment);
            } else if segment.bytes().all(|byte| byte.is_ascii_digit()) {
                display.push('[');
                display.push_str(&segment);
                display.push(']');
            } else {
                display.push('.');
                display.push_str(&segment);
            }
            display
        })
}

#[cfg(test)]
mod tests {
    use litellm_error::{Report, find};
    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::{Catalog, wire::IsoDate};

    #[rstest]
    #[case::json(Repr::Json(serde_json::from_str::<Value>("{").unwrap_err()), Rejects::Document)]
    #[case::empty(Repr::Empty, Rejects::Document)]
    #[case::below_minimum(Repr::BelowMinimum { actual: 1, minimum: 2 }, Rejects::Document)]
    #[case::shrunk(Repr::Shrunk { actual: 1, reference: 3, ratio: 0.5 }, Rejects::Document)]
    #[case::not_an_object(Repr::NotAnObject, Rejects::Row)]
    #[case::invalid(Repr::Invalid(serde_json::from_str::<Value>("x").unwrap_err()), Rejects::Row)]
    #[case::unknown_block_key(Repr::UnknownBlockKey, Rejects::Row)]
    #[case::missing_provider(Repr::MissingProvider, Rejects::Nothing)]
    #[case::unknown_timezone(Repr::UnknownTimezone { zone: "invalid".into() }, Rejects::Row)]
    #[case::alias_shadows_model(Repr::AliasShadowsModel { alias: "model".into() }, Rejects::Row)]
    #[case::alias_taken(Repr::AliasTaken { alias: "alias".into(), taken_by: "owner".into() }, Rejects::Row)]
    #[case::duplicate_price(Repr::DuplicatePrice { first: "a".into(), second: "b".into() }, Rejects::Row)]
    #[case::suffix_inside_block(Repr::SuffixInsideBlock { block: "tiered_pricing" }, Rejects::Row)]
    #[case::tier_range(Repr::TierRange, Rejects::Row)]
    #[case::tier_reversed(Repr::TierReversed, Rejects::Row)]
    #[case::tier_gap(Repr::TierGap { start: 1, expected: 0 }, Rejects::Row)]
    #[case::mixed_tier_basis(Repr::MixedTierBasis, Rejects::Row)]
    #[case::not_priced_by_results(Repr::NotPricedByResults, Rejects::Row)]
    #[case::bands_from_two_sources(Repr::BandsFromTwoSources, Rejects::Row)]
    #[case::not_an_off_peak_rate(Repr::NotAnOffPeakRate, Rejects::Row)]
    #[case::no_off_peak_prices(Repr::NoOffPeakPrices, Rejects::Row)]
    #[case::no_off_peak_hours(Repr::NoOffPeakHours, Rejects::Row)]
    #[case::empty_list(Repr::EmptyList, Rejects::Row)]
    #[case::invalid_date(Repr::InvalidDate { cause: None }, Rejects::Row)]
    #[case::unread_field(Repr::UnreadField, Rejects::Nothing)]
    #[case::no_fallback_rules(Repr::NoFallbackRules, Rejects::Nothing)]
    fn policy_classifies_each_error(#[case] error: Repr, #[case] expected: Rejects) {
        assert_eq!(Error::from(error).rejects(), expected);
    }

    #[rstest]
    #[case::root("", "x", "/x")]
    #[case::empty_model("/", "x", "//x")]
    #[case::empty_parent_segment("/m/", "x", "/m//x")]
    #[case::escaped("/m~1n", "a~/b", "/m~1n/a~0~1b")]
    fn pointers_preserve_segments(#[case] parent: &str, #[case] key: &str, #[case] expected: &str) {
        assert_eq!(pointer_child(parent, key), expected);
    }

    #[rstest]
    #[case::malformed(br#"{"#)]
    #[case::bad_price(br#"{"m":{"litellm_provider":"test","input_cost_per_token":"free"}}"#)]
    fn json_causes_are_typed_and_rendered_once(#[case] body: &[u8]) {
        let error = match Catalog::parse(body) {
            Ok(loaded) => loaded.errors.into_iter().next().expect("invalid price"),
            Err(error) => error,
        };
        let cause = find::<serde_json::Error>(&error)
            .expect("typed JSON cause")
            .to_string();
        let report = Report::new(&error).to_string();
        assert_eq!(report.matches(&cause).count(), 1, "{report}");
        assert!(!error.to_string().contains(&cause));
    }

    #[rstest]
    #[case::year("xxxx-01-01")]
    #[case::month("2026-xx-01")]
    #[case::day("2026-01-xx")]
    fn date_number_errors_keep_the_cause(#[case] value: &str) {
        let error = value.parse::<IsoDate>().unwrap_err();
        let cause = find::<ParseIntError>(&error)
            .expect("typed integer cause")
            .to_string();
        assert_eq!(error.rejects(), Rejects::Row);
        assert_eq!(Report::new(&error).to_string().matches(&cause).count(), 1);
        assert!(!error.to_string().contains(&cause));
    }

    #[rstest]
    #[case::month("2026-13-01")]
    #[case::day("2026-02-30")]
    fn date_range_errors_keep_the_cause(#[case] value: &str) {
        let error = value.parse::<IsoDate>().unwrap_err();
        let cause = find::<time::error::ComponentRange>(&error)
            .expect("typed calendar cause")
            .to_string();
        assert_eq!(error.rejects(), Rejects::Row);
        assert_eq!(Report::new(&error).to_string().matches(&cause).count(), 1);
        assert!(!error.to_string().contains(&cause));
    }
}
