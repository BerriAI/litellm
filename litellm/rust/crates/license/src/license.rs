use jiff::civil::Date;
use jiff::fmt::strtime;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use serde_with::{DefaultOnNull, OneOrMany, serde_as};

use crate::Feature;

const ALL_FEATURES: &str = "*";

/// The signed claims of a verified license. A claim of the wrong type fails verification, a null claim reads as
/// absent and an unknown claim is ignored
#[serde_as]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct License {
    #[serde(deserialize_with = "expiration_date")]
    pub expiration_date: Date,
    pub user_id: Option<String>,
    #[serde_as(as = "DefaultOnNull<OneOrMany<_>>")]
    #[serde(default)]
    pub allowed_features: Vec<String>,
    pub max_users: Option<i64>,
    pub max_teams: Option<i64>,
}

fn expiration_date<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Date, D::Error> {
    let text = String::deserialize(deserializer)?;
    strtime::parse("%Y-%m-%d", &text)
        .and_then(|parsed| parsed.to_date())
        .map_err(|error| {
            D::Error::custom(format!("expiration_date is not a YYYY-MM-DD date: {error}"))
        })
}

impl License {
    pub fn grants(&self, feature: Feature) -> bool {
        let feature: &str = feature.into();
        self.allowed_features
            .iter()
            .any(|granted| granted == feature || granted == ALL_FEATURES)
    }

    pub fn is_over_user_limit(&self, users: i64) -> bool {
        self.max_users.is_some_and(|limit| users > limit)
    }

    pub fn is_over_team_limit(&self, teams: i64) -> bool {
        self.max_teams.is_some_and(|limit| teams > limit)
    }
}

#[cfg(test)]
mod tests {
    use jiff::civil::date;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn claims(mut claims: Value) -> Result<License, serde_json::Error> {
        claims["expiration_date"] = json!("2027-11-12");
        serde_json::from_value(claims)
    }

    fn license(claims_json: Value) -> License {
        claims(claims_json).unwrap()
    }

    #[rstest]
    fn reads_every_known_claim() {
        let license = license(json!({
            "user_id": "u",
            "allowed_features": ["sso", "auto_router"],
            "max_users": 5,
            "max_teams": 2,
            "unknown_claim": true,
        }));

        assert_eq!(
            license,
            License {
                expiration_date: date(2027, 11, 12),
                user_id: Some("u".into()),
                allowed_features: vec!["sso".into(), "auto_router".into()],
                max_users: Some(5),
                max_teams: Some(2),
            }
        );
    }

    #[rstest]
    #[case::user_id_number(json!({"user_id": 7}))]
    #[case::max_users_float(json!({"max_users": 5.0}))]
    #[case::max_users_string(json!({"max_users": "5"}))]
    #[case::max_teams_bool(json!({"max_teams": true}))]
    #[case::allowed_features_number(json!({"allowed_features": 5}))]
    #[case::allowed_features_non_string_entry(json!({"allowed_features": ["sso", 1]}))]
    fn rejects_a_claim_of_another_type(#[case] malformed: Value) {
        assert!(claims(malformed).is_err());
    }

    #[rstest]
    fn reads_a_null_claim_as_absent() {
        let nulls = license(json!({
            "user_id": null,
            "allowed_features": null,
            "max_users": null,
            "max_teams": null,
        }));

        assert_eq!(nulls, license(json!({})));
    }

    #[rstest]
    #[case::listed(json!(["sso", "auto_router"]), true)]
    #[case::wildcard_in_list(json!(["sso", "*"]), true)]
    #[case::bare_wildcard(json!("*"), true)]
    #[case::bare_feature(json!("auto_router"), true)]
    #[case::other_features(json!(["sso"]), false)]
    fn grants_a_listed_feature_or_the_wildcard(#[case] features: Value, #[case] granted: bool) {
        assert_eq!(
            license(json!({"allowed_features": features})).grants(Feature::AutoRouter),
            granted
        );
    }

    #[rstest]
    fn grants_nothing_without_allowed_features() {
        assert!(!license(json!({})).grants(Feature::AutoRouter));
    }

    #[rstest]
    #[case::under(99, false)]
    #[case::at(100, false)]
    #[case::over(101, true)]
    fn limits_users_and_teams_above_their_maximum(#[case] count: i64, #[case] over: bool) {
        let license = license(json!({"max_users": 100, "max_teams": 100}));

        assert_eq!(license.is_over_user_limit(count), over);
        assert_eq!(license.is_over_team_limit(count), over);
    }

    #[rstest]
    fn unlimited_without_a_maximum() {
        let license = license(json!({}));

        assert!(!license.is_over_user_limit(i64::MAX));
        assert!(!license.is_over_team_limit(i64::MAX));
    }
}
