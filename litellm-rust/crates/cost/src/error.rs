use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error, strum::VariantArray)]
pub enum PriceEvaluationError {
    #[error("only token meters can be projected")]
    NonTokenProjection,
    #[error("pricing produced a non-finite amount or rate")]
    NonFiniteValue,
    #[error("quantity must be finite and non-negative")]
    InvalidQuantity,
    #[error("measured quantity unit is not valid for this meter")]
    InvalidQuantityUnit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidUtcWindow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidWeekday;

impl fmt::Display for InvalidUtcWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected a UTC window HH:MM-HH:MM")
    }
}

impl fmt::Display for InvalidWeekday {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("expected an ISO weekday number 1-7 or an English day name")
    }
}

impl std::error::Error for InvalidUtcWindow {}
impl std::error::Error for InvalidWeekday {}
