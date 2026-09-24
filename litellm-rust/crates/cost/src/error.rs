#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CostError {
    #[error("usage payload has an invalid shape")]
    InvalidShape,
    #[error("usage payload has an uncoercible value")]
    InvalidUsage,
    #[error("reported cost is not a finite number")]
    InvalidCost,
    #[error("duration is not a finite number of seconds")]
    InvalidDuration,
    #[error("pricing multiplier is not a usable number")]
    InvalidMultiplier,
    #[error("provider-reported cost is not a number")]
    InvalidProviderCost,
    #[error("quantity is missing or not a positive number")]
    InvalidQuantity,
    #[error("rate is negative, non-finite, or unparseable")]
    InvalidRate,
    #[error("rate basis must be per image, per token, or per pixel")]
    InvalidRateBasis,
    #[error("no input cost per character rate")]
    MissingInputCharacterRate,
    #[error("no cost metric for the model")]
    MissingMetric,
    #[error("no model in the response or request")]
    MissingModel,
    #[error("no page count in the OCR usage")]
    MissingPages,
    #[error("no prompt character count")]
    MissingPromptCharacters,
    #[error("no provider for a call that needs one")]
    MissingProvider,
    #[error("no rate for the priced unit")]
    MissingRate,
    #[error("image steps are not a number")]
    InvalidSteps,
    #[error("no usage on the response")]
    MissingUsage,
    #[error("model is not in the cost map")]
    ModelNotFound,
    #[error("computed cost is not finite")]
    NonFiniteCost,
    #[error("token count is not a usable number")]
    TokenCount,
    #[error("token counts overflow a u64")]
    TokenCountOverflow,
    #[error("provider has no image cost path")]
    UnsupportedProvider,
}
