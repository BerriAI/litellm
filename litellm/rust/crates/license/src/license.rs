use jiff::civil::Date;
use serde_json::{Map, Value};

use crate::Feature;

const ALL_FEATURES: &str = "*";

/// The signed claims of a verified license. A claim of an unexpected type reads as absent, as the proxy's
/// `isinstance` checks did
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct License {
    pub expiration_date: Date,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    pub allowed_features: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_users: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_teams: Option<i64>,
}

impl License {
    pub(crate) fn from_claims(expiration_date: Date, claims: &Map<String, Value>) -> Self {
        let integer = |name: &str| claims.get(name).and_then(Value::as_i64);
        Self {
            expiration_date,
            user_id: claims
                .get("user_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            allowed_features: match claims.get("allowed_features") {
                Some(Value::Array(features)) => features
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
                Some(Value::String(feature)) => vec![feature.clone()],
                _ => Vec::new(),
            },
            max_users: integer("max_users"),
            max_teams: integer("max_teams"),
        }
    }

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
    use serde_json::json;

    use super::*;

    fn license(claims: Value) -> License {
        License::from_claims(date(2027, 11, 12), claims.as_object().unwrap())
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
    #[case::max_teams_null(json!({"max_teams": null}))]
    fn reads_a_claim_of_another_type_as_absent(#[case] claims: Value) {
        let license = license(claims);

        assert_eq!(
            (license.user_id, license.max_users, license.max_teams),
            (None, None, None)
        );
    }

    #[rstest]
    #[case::listed(json!(["sso", "auto_router"]), true)]
    #[case::wildcard_in_list(json!(["sso", "*"]), true)]
    #[case::bare_wildcard(json!("*"), true)]
    #[case::bare_feature(json!("auto_router"), true)]
    #[case::other_features(json!(["sso"]), false)]
    #[case::non_string_entries(json!([1, null, {"auto_router": true}]), false)]
    #[case::number(json!(5), false)]
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
    fn serializes_to_the_claims_the_proxy_reads() {
        let license = license(
            json!({"user_id": "u", "allowed_features": ["sso"], "max_users": 5, "dropped": 1}),
        );

        assert_eq!(
            serde_json::to_value(license).unwrap(),
            json!({"expiration_date": "2027-11-12", "user_id": "u", "allowed_features": ["sso"], "max_users": 5})
        );
    }

    #[rstest]
    fn unlimited_without_a_maximum() {
        let license = license(json!({}));

        assert!(!license.is_over_user_limit(i64::MAX));
        assert!(!license.is_over_team_limit(i64::MAX));
    }
}
