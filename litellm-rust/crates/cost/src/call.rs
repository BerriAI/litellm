use crate::price::{Channel, Prices, Residency, ServiceTier};

/// Facts about the call, never about the row or the provider
#[derive(Clone, Debug, PartialEq)]
pub struct CallContext {
    /// Requested service tier, `"fast"` maps to `Priority` as on main
    pub tier: ServiceTier,
    /// Online, batch or decisions
    pub channel: Channel,
    /// OpenAI data residency, picks the residency uplift
    pub residency: Option<Residency>,
    /// Vertex call sent to a regional endpoint, picks the regional uplift
    pub non_global_endpoint: bool,
    /// Anthropic `speed: fast`, picks the speed uplift
    pub speed: Option<Speed>,
    /// Anthropic and Bedrock `inference_geo`, picks the geo uplift
    pub inference_geo: Option<InferenceGeo>,
    /// When the call was billed, for off-peak windows
    pub at: jiff::Timestamp,
    /// Per-call custom pricing, lowered like a row and applied field by field over it
    pub overlay: Option<Prices>,
}

/// Anthropic `speed`
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Speed {
    Fast,
}

/// Anthropic and Bedrock `inference_geo`, lowercased as on main
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InferenceGeo(Box<str>);

impl InferenceGeo {
    pub fn new(geo: &str) -> Self {
        Self(geo.to_lowercase().into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}
