use crate::Raw;

#[macro_rules_attribute::apply(crate::config_section)]
pub struct FallbackSettings {
    pub default_fallbacks: Option<Raw>,
    pub context_window_fallbacks: Option<Raw>,
}
