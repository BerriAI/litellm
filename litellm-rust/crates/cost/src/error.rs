use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostError {
    InvalidShape,
    InvalidUsage,
    InvalidCost,
    InvalidDuration,
    InvalidMultiplier,
    InvalidProviderCost,
    InvalidQuantity,
    InvalidRate,
    InvalidRateBasis,
    MissingInputCharacterRate,
    MissingMetric,
    MissingModel,
    MissingPages,
    MissingPromptCharacters,
    MissingProvider,
    MissingRate,
    MissingUsage,
    ModelNotFound,
    NonFiniteCost,
    TokenCount,
    TokenCountOverflow,
    UnsupportedCallType,
    UnsupportedProvider,
}

impl fmt::Display for CostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::InvalidShape => "usage payload has an invalid shape",
            Self::InvalidUsage => "usage payload has an uncoercible value",
            Self::InvalidCost => "reported cost is not a finite number",
            Self::InvalidDuration => "duration is not a finite number of seconds",
            Self::InvalidMultiplier => "pricing multiplier is not a usable number",
            Self::InvalidProviderCost => "provider-reported cost is not a number",
            Self::InvalidQuantity => "quantity is missing or not a positive number",
            Self::InvalidRate => "rate is negative, non-finite, or unparseable",
            Self::InvalidRateBasis => "rate basis must be per image, per token, or per pixel",
            Self::MissingInputCharacterRate => "no input cost per character rate",
            Self::MissingMetric => "no cost metric for the model",
            Self::MissingModel => "no model in the response or request",
            Self::MissingPages => "no page count in the OCR usage",
            Self::MissingPromptCharacters => "no prompt character count",
            Self::MissingProvider => "no provider for a call that needs one",
            Self::MissingRate => "no rate for the priced unit",
            Self::MissingUsage => "no usage on the response",
            Self::ModelNotFound => "model is not in the cost map",
            Self::NonFiniteCost => "computed cost is not finite",
            Self::TokenCount => "token count is not a usable number",
            Self::TokenCountOverflow => "token counts overflow a u64",
            Self::UnsupportedCallType => "call type has no cost path",
            Self::UnsupportedProvider => "provider has no image cost path",
        };
        f.write_str(text)
    }
}
