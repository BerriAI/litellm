use litellm_core_utils::settings::{Lookup, ProcessEnvironment};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportedEffortTiers {
    #[serde(default)]
    pub minimal: bool,
    #[serde(default)]
    pub low: bool,
    #[serde(default)]
    pub medium: bool,
    #[serde(default)]
    pub high: bool,
    #[serde(default)]
    pub xhigh: bool,
    #[serde(default)]
    pub max: bool,
}

impl SupportedEffortTiers {
    pub fn any(self) -> bool {
        self.minimal || self.low || self.medium || self.high || self.xhigh || self.max
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessagesModelCapabilities {
    #[serde(default)]
    pub supports_reasoning: bool,
    #[serde(default)]
    pub supports_adaptive_thinking: bool,
    #[serde(default)]
    pub thinking_always_on: bool,
    #[serde(default)]
    pub supports_legacy_thinking: bool,
    #[serde(default)]
    pub supports_output_config: bool,
    #[serde(default = "default_true")]
    pub supports_sampling_params: bool,
    #[serde(default)]
    pub supports_speed: bool,
    #[serde(default)]
    pub effort_tiers: SupportedEffortTiers,
}

fn default_true() -> bool {
    true
}

impl Default for MessagesModelCapabilities {
    fn default() -> Self {
        Self {
            supports_reasoning: false,
            supports_adaptive_thinking: false,
            thinking_always_on: false,
            supports_legacy_thinking: false,
            supports_output_config: false,
            supports_sampling_params: true,
            supports_speed: false,
            effort_tiers: SupportedEffortTiers::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkingBudgets {
    pub minimal: u64,
    pub low: u64,
    pub medium: u64,
    pub high: u64,
    pub xhigh: u64,
    pub max: u64,
}

impl Default for ThinkingBudgets {
    fn default() -> Self {
        Self {
            minimal: 128,
            low: 1024,
            medium: 2048,
            high: 4096,
            xhigh: 8192,
            max: 16384,
        }
    }
}

impl ThinkingBudgets {
    pub fn from_lookup(env: &impl Lookup) -> Self {
        let defaults = Self::default();
        let tier = |name: &str, default: u64| {
            env.parsed::<u64>(&format!("DEFAULT_REASONING_EFFORT_{name}_THINKING_BUDGET"))
                .unwrap_or(default)
        };
        Self {
            minimal: tier("MINIMAL", defaults.minimal),
            low: tier("LOW", defaults.low),
            medium: tier("MEDIUM", defaults.medium),
            high: tier("HIGH", defaults.high),
            xhigh: tier("XHIGH", defaults.xhigh),
            max: tier("MAX", defaults.max),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThinkingContext {
    pub capabilities: MessagesModelCapabilities,
    pub budgets: ThinkingBudgets,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MessagesTransformContext {
    pub thinking: ThinkingContext,
    pub drop_params: bool,
}
impl MessagesTransformContext {
    pub fn new(capabilities: MessagesModelCapabilities, drop_params: bool) -> Self {
        Self::with_lookup(capabilities, drop_params, &ProcessEnvironment)
    }

    pub fn with_lookup(
        capabilities: MessagesModelCapabilities,
        drop_params: bool,
        env: &impl Lookup,
    ) -> Self {
        Self {
            thinking: ThinkingContext {
                capabilities,
                budgets: ThinkingBudgets::from_lookup(env),
            },
            drop_params,
        }
    }
}
