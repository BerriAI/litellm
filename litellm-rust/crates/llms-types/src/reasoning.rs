use strum::IntoStaticStr;

/// Reasoning effort level accepted or applied by the model.
#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq, IntoStaticStr)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub const ALL: [Self; 7] = [
        Self::None,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    pub fn as_str(self) -> &'static str {
        self.into()
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|effort| effort.as_str() == value)
    }
}

#[cfg(test)]
mod tests {
    use super::ReasoningEffort;
    use rstest::rstest;
    use serde_json::Value;

    #[rstest]
    fn reasoning_effort_names_match_the_wire_and_parse_back(
        #[values(
            ReasoningEffort::None,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max
        )]
        effort: ReasoningEffort,
    ) {
        assert_eq!(
            serde_json::to_value(effort).unwrap(),
            Value::String(effort.as_str().to_string())
        );
        assert_eq!(ReasoningEffort::parse(effort.as_str()), Some(effort));
        assert!(ReasoningEffort::ALL.contains(&effort));
    }

    #[rstest]
    #[case::unknown("ultra")]
    #[case::uppercase("HIGH")]
    #[case::empty("")]
    fn reasoning_effort_parse_rejects(#[case] value: &str) {
        assert_eq!(ReasoningEffort::parse(value), None);
    }
}
