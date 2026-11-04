use litellm_model_catalog::IntegrityLimits;
use litellm_model_catalog_refresh::DEFAULT_REGISTRY_URL;
use litellm_python_compat::BigInt;

const DEFAULT_MIN_MODEL_COUNT: usize = 50;
const DEFAULT_MAX_SHRINK_RATIO: f64 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// `None` when `LITELLM_LOCAL_MODEL_COST_MAP` or offline mode keeps the bundled map
    pub remote_url: Option<String>,
    pub limits: IntegrityLimits,
}

impl Settings {
    /// The variables Python reads in `get_model_cost_map.py` and `constants.py`, parsed the way it parses them
    pub fn from_env(env: impl Fn(&str) -> Option<String>) -> Self {
        let offline = env("LITELLM_OFFLINE_MODE")
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("true"));
        let local = env("LITELLM_LOCAL_MODEL_COST_MAP")
            .is_some_and(|value| value.eq_ignore_ascii_case("true"));
        let remote_url = (!offline && !local).then(|| {
            env("LITELLM_MODEL_COST_MAP_URL").unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_owned())
        });
        Self {
            remote_url,
            limits: limits(&env),
        }
    }
}

fn limits(env: &impl Fn(&str) -> Option<String>) -> IntegrityLimits {
    let min_model_count = env("MODEL_COST_MAP_MIN_MODEL_COUNT")
        .as_deref()
        .and_then(litellm_python_compat::number::int)
        .map_or(DEFAULT_MIN_MODEL_COUNT, |count| saturating_count(&count));
    let ratio = env("MODEL_COST_MAP_MAX_SHRINK_RATIO")
        .as_deref()
        .and_then(litellm_python_compat::number::float)
        .unwrap_or(DEFAULT_MAX_SHRINK_RATIO);
    IntegrityLimits::new(min_model_count, ratio).unwrap_or_else(|| {
        tracing::warn!(
            ratio,
            "MODEL_COST_MAP_MAX_SHRINK_RATIO must be between 0 and 1, using 0.5"
        );
        IntegrityLimits::new(min_model_count, DEFAULT_MAX_SHRINK_RATIO)
            .expect("0.5 is a valid ratio")
    })
}

/// A negative minimum passes every map, as it does in Python
fn saturating_count(count: &BigInt) -> usize {
    if *count < BigInt::from(0) {
        return 0;
    }
    usize::try_from(count).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn settings(variables: &[(&str, &str)]) -> Settings {
        Settings::from_env(|name| {
            variables
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        })
    }

    fn limits(min_model_count: usize, ratio: f64) -> IntegrityLimits {
        IntegrityLimits::new(min_model_count, ratio).unwrap()
    }

    #[rstest]
    #[case::default(&[], Some(DEFAULT_REGISTRY_URL))]
    #[case::custom_url(&[("LITELLM_MODEL_COST_MAP_URL", "file:///tmp/map.json")], Some("file:///tmp/map.json"))]
    #[case::empty_url_is_still_fetched(&[("LITELLM_MODEL_COST_MAP_URL", "")], Some(""))]
    #[case::local(&[("LITELLM_LOCAL_MODEL_COST_MAP", "TRUE")], None)]
    #[case::local_is_not_stripped(&[("LITELLM_LOCAL_MODEL_COST_MAP", " true")], Some(DEFAULT_REGISTRY_URL))]
    #[case::local_other_value(&[("LITELLM_LOCAL_MODEL_COST_MAP", "1")], Some(DEFAULT_REGISTRY_URL))]
    #[case::offline(&[("LITELLM_OFFLINE_MODE", " True ")], None)]
    #[case::offline_wins_over_url(&[("LITELLM_OFFLINE_MODE", "true"), ("LITELLM_MODEL_COST_MAP_URL", "s3://b/k")], None)]
    fn the_source_follows_python_s_variables(
        #[case] variables: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(settings(variables).remote_url.as_deref(), expected);
    }

    #[rstest]
    #[case::defaults(&[], limits(50, 0.5))]
    #[case::configured(&[("MODEL_COST_MAP_MIN_MODEL_COUNT", " 1_0 "), ("MODEL_COST_MAP_MAX_SHRINK_RATIO", "0.25")], limits(10, 0.25))]
    #[case::negative_minimum(&[("MODEL_COST_MAP_MIN_MODEL_COUNT", "-5")], limits(0, 0.5))]
    #[case::huge_minimum(&[("MODEL_COST_MAP_MIN_MODEL_COUNT", "99999999999999999999999")], limits(usize::MAX, 0.5))]
    #[case::unparsable_minimum(&[("MODEL_COST_MAP_MIN_MODEL_COUNT", "ten")], limits(50, 0.5))]
    #[case::blank_ratio(&[("MODEL_COST_MAP_MAX_SHRINK_RATIO", "  ")], limits(50, 0.5))]
    #[case::ratio_above_one(&[("MODEL_COST_MAP_MAX_SHRINK_RATIO", "1.5")], limits(50, 0.5))]
    fn limits_parse_like_get_env_int_and_float(
        #[case] variables: &[(&str, &str)],
        #[case] expected: IntegrityLimits,
    ) {
        assert_eq!(settings(variables).limits, expected);
    }
}
