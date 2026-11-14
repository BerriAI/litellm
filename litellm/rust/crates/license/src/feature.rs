/// A capability a signed license can grant by listing it in `allowed_features`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr, strum::EnumString)]
pub enum Feature {
    #[strum(serialize = "auto_router")]
    AutoRouter,
}

impl Feature {
    pub fn remedy(self) -> String {
        format!(
            "A LiteLLM license with the '{}' feature lifts the limit.",
            <&'static str>::from(self)
        )
    }
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

    #[rstest]
    fn remedy_names_the_license_feature() {
        assert_eq!(
            Feature::AutoRouter.remedy(),
            "A LiteLLM license with the 'auto_router' feature lifts the limit."
        );
    }
}
