//! One entry of a caller's `models` list, as the proxy spells it on a key, team, user or org row.
//! Authz and the router both read a list through it, so no reader compares an entry with a
//! special spelling itself.

/// An entry that names no model: the proxy's `SpecialModelNames`, plus `*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString)]
pub enum SpecialModel {
    #[strum(serialize = "*", serialize = "all-proxy-models")]
    Every,
    #[strum(serialize = "all-team-models")]
    TeamModels,
    #[strum(serialize = "no-default-models")]
    NoDefaultModels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelEntry<'a> {
    Special(SpecialModel),
    /// A wildcard the proxy matches against a model's provider, such as `anthropic/*`.
    Pattern(&'a str),
    /// A model name, an alias or an access group. Only the router's registry tells them apart.
    Name(&'a str),
}

impl<'a> ModelEntry<'a> {
    pub fn of(entry: &'a str) -> Self {
        match entry.parse() {
            Ok(special) => Self::Special(special),
            Err(_) if entry.contains('*') => Self::Pattern(entry),
            Err(_) => Self::Name(entry),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::a_star("*", ModelEntry::Special(SpecialModel::Every))]
    #[case::every_proxy_model("all-proxy-models", ModelEntry::Special(SpecialModel::Every))]
    #[case::the_teams_models("all-team-models", ModelEntry::Special(SpecialModel::TeamModels))]
    #[case::no_default_models(
        "no-default-models",
        ModelEntry::Special(SpecialModel::NoDefaultModels)
    )]
    #[case::a_provider_pattern("anthropic/*", ModelEntry::Pattern("anthropic/*"))]
    #[case::a_name("gpt", ModelEntry::Name("gpt"))]
    #[case::a_special_spelling_in_another_case(
        "All-Proxy-Models",
        ModelEntry::Name("All-Proxy-Models")
    )]
    #[case::an_empty_entry("", ModelEntry::Name(""))]
    fn an_entry_is_read_by_its_spelling(#[case] entry: &str, #[case] read: ModelEntry<'_>) {
        assert_eq!(ModelEntry::of(entry), read);
    }
}
