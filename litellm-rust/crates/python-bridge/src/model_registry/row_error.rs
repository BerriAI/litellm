use litellm_model_catalog::{Error, Located};
use pyo3::prelude::*;
use strum::{IntoStaticStr, VariantArray};

#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum RowErrorKind {
    #[strum(serialize = "json")]
    Json,
    #[strum(serialize = "empty")]
    Empty,
    #[strum(serialize = "below_minimum")]
    BelowMinimum,
    #[strum(serialize = "shrunk")]
    Shrunk,
    #[strum(serialize = "not_an_object")]
    NotAnObject,
    #[strum(serialize = "invalid")]
    Invalid,
    #[strum(serialize = "unknown_block_key")]
    UnknownBlockKey,
    #[strum(serialize = "missing_provider")]
    MissingProvider,
    #[strum(serialize = "unknown_timezone")]
    UnknownTimezone,
    #[strum(serialize = "alias_shadows_model")]
    AliasShadowsModel,
    #[strum(serialize = "alias_taken")]
    AliasTaken,
    #[strum(serialize = "duplicate_price")]
    DuplicatePrice,
    #[strum(serialize = "suffix_inside_block")]
    SuffixInsideBlock,
    #[strum(serialize = "tier_range")]
    TierRange,
    #[strum(serialize = "tier_reversed")]
    TierReversed,
    #[strum(serialize = "tier_gap")]
    TierGap,
    #[strum(serialize = "mixed_tier_basis")]
    MixedTierBasis,
    #[strum(serialize = "not_priced_by_results")]
    NotPricedByResults,
    #[strum(serialize = "bands_from_two_sources")]
    BandsFromTwoSources,
    #[strum(serialize = "not_an_off_peak_rate")]
    NotAnOffPeakRate,
    #[strum(serialize = "no_off_peak_prices")]
    NoOffPeakPrices,
    #[strum(serialize = "no_off_peak_hours")]
    NoOffPeakHours,
    #[strum(serialize = "empty_list")]
    EmptyList,
    #[strum(serialize = "invalid_date")]
    InvalidDate,
}

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone, Debug)]
pub(crate) struct RowError {
    #[pyo3(get)]
    pub(crate) key: String,
    #[pyo3(get)]
    pub(crate) field: Option<String>,
    #[pyo3(get)]
    pub(crate) kind: String,
    #[pyo3(get)]
    pub(crate) rejects: bool,
    #[pyo3(get)]
    pub(crate) message: String,
}

pub(crate) fn row_error(error: &Located) -> RowError {
    let mut path = error.path.split('/').skip(1);
    let key = unescape_pointer_segment(path.next().unwrap_or_default());
    let field = path.map(unescape_pointer_segment).collect::<Vec<_>>();
    let field = (!field.is_empty()).then(|| field.join("/"));
    RowError {
        key,
        field,
        kind: <&'static str>::from(error_kind(&error.error)).to_owned(),
        rejects: litellm_model_catalog::rejects_row(&error.error),
        message: error.error.to_string(),
    }
}

fn unescape_pointer_segment(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

fn error_kind(error: &Error) -> RowErrorKind {
    match error {
        Error::Json(_) => RowErrorKind::Json,
        Error::Empty => RowErrorKind::Empty,
        Error::BelowMinimum { .. } => RowErrorKind::BelowMinimum,
        Error::Shrunk { .. } => RowErrorKind::Shrunk,
        Error::NotAnObject => RowErrorKind::NotAnObject,
        Error::Invalid(_) => RowErrorKind::Invalid,
        Error::UnknownBlockKey => RowErrorKind::UnknownBlockKey,
        Error::MissingProvider => RowErrorKind::MissingProvider,
        Error::UnknownTimezone { .. } => RowErrorKind::UnknownTimezone,
        Error::AliasShadowsModel { .. } => RowErrorKind::AliasShadowsModel,
        Error::AliasTaken { .. } => RowErrorKind::AliasTaken,
        Error::DuplicatePrice { .. } => RowErrorKind::DuplicatePrice,
        Error::SuffixInsideBlock { .. } => RowErrorKind::SuffixInsideBlock,
        Error::TierRange => RowErrorKind::TierRange,
        Error::TierReversed => RowErrorKind::TierReversed,
        Error::TierGap { .. } => RowErrorKind::TierGap,
        Error::MixedTierBasis => RowErrorKind::MixedTierBasis,
        Error::NotPricedByResults => RowErrorKind::NotPricedByResults,
        Error::BandsFromTwoSources => RowErrorKind::BandsFromTwoSources,
        Error::NotAnOffPeakRate => RowErrorKind::NotAnOffPeakRate,
        Error::NoOffPeakPrices => RowErrorKind::NoOffPeakPrices,
        Error::NoOffPeakHours => RowErrorKind::NoOffPeakHours,
        Error::EmptyList => RowErrorKind::EmptyList,
        Error::InvalidDate => RowErrorKind::InvalidDate,
    }
}
