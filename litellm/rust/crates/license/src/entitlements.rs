use crate::{Feature, License};

/// What the configured license lets this deployment do
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entitlements {
    Community,
    Signed(License),
    ServerVerified,
}

impl Entitlements {
    pub fn is_enterprise(&self) -> bool {
        !matches!(self, Self::Community)
    }

    pub fn license(&self) -> Option<&License> {
        match self {
            Self::Signed(license) => Some(license),
            Self::ServerVerified | Self::Community => None,
        }
    }

    pub fn grants(&self, feature: Feature) -> bool {
        self.license()
            .is_some_and(|license| license.grants(feature))
    }

    pub fn max_users(&self) -> Option<i64> {
        self.license().and_then(|license| license.max_users)
    }

    pub fn is_over_user_limit(&self, users: i64) -> bool {
        self.license()
            .is_some_and(|license| license.is_over_user_limit(users))
    }

    pub fn is_over_team_limit(&self, teams: i64) -> bool {
        self.license()
            .is_some_and(|license| license.is_over_team_limit(teams))
    }
}

#[cfg(test)]
mod tests {
    use jiff::civil::date;
    use rstest::rstest;

    use super::*;

    fn signed() -> Entitlements {
        Entitlements::Signed(License {
            expiration_date: date(2027, 11, 12),
            user_id: None,
            allowed_features: vec!["auto_router".into()],
            max_users: Some(10),
            max_teams: Some(2),
        })
    }

    #[rstest]
    #[case::community(Entitlements::Community, false)]
    #[case::server_verified(Entitlements::ServerVerified, true)]
    #[case::signed(signed(), true)]
    fn any_verified_license_is_enterprise(
        #[case] entitlements: Entitlements,
        #[case] enterprise: bool,
    ) {
        assert_eq!(entitlements.is_enterprise(), enterprise);
    }

    #[rstest]
    #[case::community(Entitlements::Community, false)]
    #[case::server_verified(Entitlements::ServerVerified, false)]
    #[case::signed(signed(), true)]
    fn only_signed_claims_grant_features(
        #[case] entitlements: Entitlements,
        #[case] granted: bool,
    ) {
        assert_eq!(entitlements.grants(Feature::AutoRouter), granted);
    }

    #[rstest]
    #[case::community(Entitlements::Community, None, false, false)]
    #[case::server_verified(Entitlements::ServerVerified, None, false, false)]
    #[case::signed(signed(), Some(10), true, true)]
    fn limits_come_from_signed_claims(
        #[case] entitlements: Entitlements,
        #[case] max_users: Option<i64>,
        #[case] users_over: bool,
        #[case] teams_over: bool,
    ) {
        assert_eq!(entitlements.max_users(), max_users);
        assert_eq!(entitlements.is_over_user_limit(11), users_over);
        assert_eq!(entitlements.is_over_team_limit(3), teams_over);
    }
}
