use std::fmt;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid JSON: {0}")]
    Json(#[source] serde_json::Error),
    #[error("catalog has no models")]
    Empty,
    #[error("catalog has {actual} models, below minimum {minimum}")]
    BelowMinimum { actual: usize, minimum: usize },
    #[error("catalog has {actual} models, below {ratio} of reference count {reference}")]
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
    #[error("must be a JSON object")]
    NotAnObject,
    #[error("{0}")]
    Invalid(#[source] serde_json::Error),
    #[error("not a field this block reads")]
    UnknownBlockKey,
    #[error("litellm_provider is required")]
    MissingProvider,
    #[error("{zone:?} is not an IANA time zone")]
    UnknownTimezone { zone: String },
    #[error("alias {alias:?} is already a model name")]
    AliasShadowsModel { alias: String },
    #[error("alias {alias:?} already belongs to {taken_by}")]
    AliasTaken { alias: String, taken_by: String },
    #[error("{first} and {second} price the same meter under the same conditions")]
    DuplicatePrice { first: String, second: String },
    #[error("a price inside {block} takes no band or tier suffix")]
    SuffixInsideBlock { block: &'static str },
    #[error("a tier needs exactly one of range and max_results_range")]
    TierRange,
    #[error("the range ends below its start")]
    TierReversed,
    #[error("starts at {start}, but the tiers before it end at {expected}")]
    TierGap { start: u64, expected: u64 },
    #[error("tiered_pricing mixes token ranges and result ranges")]
    MixedTierBasis,
    #[error("a tier by result count only prices input_cost_per_query")]
    NotPricedByResults,
    #[error("tiered_pricing and prompt token bands are set in two places")]
    BandsFromTwoSources,
    #[error("has no off-peak rate")]
    NotAnOffPeakRate,
    #[error("off_peak_pricing has no prices")]
    NoOffPeakPrices,
    #[error("off_peak_pricing needs hours_utc or windows")]
    NoOffPeakHours,
    #[error("must list at least one entry")]
    EmptyList,
    #[error("expected a calendar date YYYY-MM-DD")]
    InvalidDate,
}

#[derive(Debug)]
pub struct Located {
    pub path: String,
    pub error: Error,
}

#[derive(Debug)]
pub struct Loaded<T> {
    pub value: T,
    pub errors: Vec<Located>,
}

impl Located {
    pub fn new(path: impl Into<String>, error: Error) -> Self {
        Self {
            path: path.into(),
            error,
        }
    }

    pub(crate) fn from_key(key: &str, error: Error) -> Self {
        Self::new(pointer_for_key(key), error)
    }

    pub(crate) fn from_path(path: &str, error: Error) -> Self {
        Self::new(path_to_pointer(path), error)
    }

    pub(crate) fn prepend(&mut self, key: &str) {
        let prefix = escape_pointer_segment(key);
        self.path = if self.path.is_empty() {
            format!("/{prefix}")
        } else {
            format!("/{prefix}{}", self.path)
        };
    }
}

impl fmt::Display for Located {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            return self.error.fmt(formatter);
        }
        let path = display_path(&self.path);
        match &self.error {
            Error::MissingProvider => write!(formatter, "{path}.litellm_provider is required"),
            Error::NotAnObject => write!(formatter, "{path} must be a JSON object"),
            Error::UnknownBlockKey => write!(formatter, "{path} is not a field this block reads"),
            _ => write!(formatter, "{path}: {}", self.error),
        }
    }
}

impl std::error::Error for Located {}

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
    format!(
        "{}/{}",
        parent.trim_end_matches('/'),
        escape_pointer_segment(key)
    )
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
