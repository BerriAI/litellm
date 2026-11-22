/// A capability a signed license can grant by listing it in `allowed_features`
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::EnumString, strum::Display,
)]
pub enum Feature {
    #[strum(serialize = "auto_router")]
    AutoRouter,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::auto_router("auto_router", Some(Feature::AutoRouter))]
    #[case::unknown("sso", None)]
    #[case::wrong_case("AUTO_ROUTER", None)]
    fn parses_only_license_spellings(#[case] name: &str, #[case] feature: Option<Feature>) {
        assert_eq!(name.parse::<Feature>().ok(), feature);
    }
}
