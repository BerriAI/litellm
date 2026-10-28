#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum Error {
    #[error("model has no input rate")]
    MissingInputRate,
    #[error("model has no output rate")]
    MissingOutputRate,
    #[error("rate must be finite and nonnegative")]
    InvalidRate,
    #[error("region multiplier must be finite and positive")]
    InvalidRegionMultiplier,
    #[error("off-peak pricing needs a billing minute within the day")]
    InvalidBillingTime,
    #[error("off-peak window must start before it ends within the day")]
    InvalidOffPeakWindow,
    #[error("cached tokens exceed prompt tokens")]
    CacheExceedsPrompt,
    #[error("cache write details do not add up to the cache write tokens")]
    InvalidCacheWriteDetails,
    #[error("token count or cost overflows")]
    TokenCountOverflow,
    #[error("service tier is repeated or cannot carry its own rates")]
    DuplicateTier,
    #[error("prompt token threshold is repeated")]
    DuplicateThreshold,
    #[error("service tier is repeated or cannot carry its own rates within a threshold")]
    DuplicateThresholdTier,
}
