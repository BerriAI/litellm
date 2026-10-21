use thiserror::Error;

#[derive(Debug, Error)]
pub enum InvalidDocument {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("catalog has no models")]
    Empty,
}

#[derive(Debug, Error, PartialEq)]
pub enum IntegrityViolation {
    #[error("catalog has {actual} models, below minimum {minimum}")]
    BelowMinimum { actual: usize, minimum: usize },
    #[error("catalog has {actual} models, below {ratio} of reference count {reference}")]
    Shrunk {
        actual: usize,
        reference: usize,
        ratio: f64,
    },
}

#[derive(Debug, Error)]
pub enum Problem {
    #[error("{model} must be a JSON object")]
    NotAnObject { model: String },
    #[error("{model}.{problem}")]
    Entry {
        model: String,
        #[source]
        problem: EntryProblem,
    },
    #[error("{model}.off_peak_pricing.weekday_timezone: {zone:?} is not an IANA time zone")]
    UnknownTimezone { model: String, zone: String },
    #[error("{model}: alias {alias:?} is already a model name")]
    AliasShadowsModel { model: String, alias: String },
    #[error("{model}: alias {alias:?} already belongs to {taken_by}")]
    AliasTaken {
        model: String,
        alias: String,
        taken_by: String,
    },
}

#[derive(Debug, Error)]
pub enum EntryProblem {
    #[error("{key}: {error}")]
    Invalid {
        key: String,
        #[source]
        error: serde_json::Error,
    },
    #[error("{key} is not a field this block reads")]
    UnknownBlockKey { key: String },
    #[error("litellm_provider is required")]
    MissingProvider,
    #[error(transparent)]
    Price(#[from] PriceProblem),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PriceProblem {
    #[error("{first} and {second} price the same meter under the same conditions")]
    Duplicate { first: String, second: String },
    #[error("{key}: a price inside {block} takes no band or tier suffix")]
    SuffixInsideBlock { key: String, block: &'static str },
    #[error("{row}: a tier needs exactly one of range and max_results_range")]
    TierRange { row: String },
    #[error("{row}: the range ends below its start")]
    TierReversed { row: String },
    #[error("{row}: starts at {start}, but the tiers before it end at {expected}")]
    TierGap {
        row: String,
        start: u64,
        expected: u64,
    },
    #[error("tiered_pricing mixes token ranges and result ranges")]
    MixedTierBasis,
    #[error("{key}: a tier by result count only prices input_cost_per_query")]
    NotPricedByResults { key: String },
    #[error("tiered_pricing and {key} both set prompt token bands")]
    BandsFromTwoSources { key: String },
    #[error("{key} has no off-peak rate")]
    NotAnOffPeakRate { key: String },
    #[error("off_peak_pricing has no prices")]
    NoOffPeakPrices,
    #[error("off_peak_pricing needs hours_utc or windows")]
    NoOffPeakHours,
    #[error("{key} must list at least one entry")]
    EmptyList { key: String },
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq, Hash)]
#[error("expected a calendar date YYYY-MM-DD")]
pub struct InvalidDate;
