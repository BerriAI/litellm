use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct OrganizationFacts {
    pub token_present: bool,
    pub client_present: bool,
    pub team_present: bool,
    pub token_id: Option<bool>,
    pub team_id: Option<bool>,
    pub row_present: Option<bool>,
    pub lookup_failed: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum OrganizationAction {
    Skip,
    ReadTokenId,
    ReadTeamId,
    ReadOrganization,
    UseOrganization,
}

pub fn organization(facts: &OrganizationFacts) -> OrganizationAction {
    use OrganizationAction::*;
    if !facts.token_present || !facts.client_present {
        return Skip;
    }
    match facts.token_id {
        None => return ReadTokenId,
        Some(false) if !facts.team_present => return Skip,
        Some(false) => match facts.team_id {
            None => return ReadTeamId,
            Some(false) => return Skip,
            Some(true) => {}
        },
        Some(true) => {}
    }
    if facts.lookup_failed {
        return Skip;
    }
    match facts.row_present {
        None => ReadOrganization,
        Some(false) => Skip,
        Some(true) => UseOrganization,
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MemberFacts {
    pub team_present: bool,
    pub token_present: bool,
    pub team_id: Option<bool>,
    pub user_id: Option<bool>,
    pub membership_loaded: bool,
    pub limits_read: bool,
    pub maximum_present: bool,
    pub default_id_string: bool,
    pub default_read: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum MemberAction {
    Skip,
    ReadTeamId,
    ReadUserId,
    ReadMembership,
    ReadLimits,
    ReadDefault,
    UseLimits,
}

pub fn member(facts: &MemberFacts) -> MemberAction {
    use MemberAction::*;
    if !facts.team_present {
        return Skip;
    }
    match facts.team_id {
        None => return ReadTeamId,
        Some(false) => return Skip,
        Some(true) => {}
    }
    if !facts.token_present {
        return Skip;
    }
    match facts.user_id {
        None => return ReadUserId,
        Some(false) => return Skip,
        Some(true) => {}
    }
    if !facts.membership_loaded {
        return ReadMembership;
    }
    if !facts.limits_read {
        return ReadLimits;
    }
    if !facts.maximum_present && facts.default_id_string && !facts.default_read {
        return ReadDefault;
    }
    UseLimits
}

#[derive(Clone, Debug, Deserialize)]
pub enum EmailFacts {
    Text(String),
    List(Vec<Option<String>>),
    Other,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SoftFacts {
    pub limit: Option<f64>,
    pub spend: Option<f64>,
    pub token_present: bool,
}

pub fn read_soft_emails(facts: &SoftFacts) -> bool {
    facts.token_present
        && facts
            .limit
            .zip(facts.spend)
            .is_some_and(|(limit, spend)| spend >= limit)
}

fn stripped(value: &str) -> &str {
    value.trim_matches(|character: char| {
        character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
    })
}

pub fn emails(facts: &EmailFacts) -> Vec<String> {
    match facts {
        EmailFacts::Text(value) => value
            .split(',')
            .map(stripped)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect(),
        EmailFacts::List(values) => values
            .iter()
            .flatten()
            .filter(|value| !stripped(value).is_empty())
            .cloned()
            .collect(),
        EmailFacts::Other => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::below(Some(1.0), Some(0.0), true, false)]
    #[case::at_limit(Some(1.0), Some(1.0), true, true)]
    #[case::no_token(Some(1.0), Some(2.0), false, false)]
    #[case::no_limit(None, Some(2.0), true, false)]
    #[case::nan(Some(f64::NAN), Some(2.0), true, false)]
    fn lazy_soft_metadata(
        #[case] limit: Option<f64>,
        #[case] spend: Option<f64>,
        #[case] token_present: bool,
        #[case] expected: bool,
    ) {
        assert_eq!(
            read_soft_emails(&SoftFacts {
                limit,
                spend,
                token_present
            }),
            expected
        );
    }

    #[rstest]
    fn failed_organization_lookup_is_skipped() {
        assert_eq!(
            organization(&OrganizationFacts {
                token_present: true,
                client_present: true,
                token_id: Some(true),
                lookup_failed: true,
                ..Default::default()
            }),
            OrganizationAction::Skip
        );
    }

    #[rstest]
    #[case::missing_token(false, true, None, None, OrganizationAction::Skip)]
    #[case::missing_client(true, false, None, None, OrganizationAction::Skip)]
    #[case::token_first(true, true, None, None, OrganizationAction::ReadTokenId)]
    #[case::token_selected(true, true, Some(true), None, OrganizationAction::ReadOrganization)]
    #[case::missing_row(true, true, Some(true), Some(false), OrganizationAction::Skip)]
    #[case::loaded_row(
        true,
        true,
        Some(true),
        Some(true),
        OrganizationAction::UseOrganization
    )]
    fn organization_stages(
        #[case] token_present: bool,
        #[case] client_present: bool,
        #[case] token_id: Option<bool>,
        #[case] row_present: Option<bool>,
        #[case] expected: OrganizationAction,
    ) {
        assert_eq!(
            organization(&OrganizationFacts {
                token_present,
                client_present,
                token_id,
                row_present,
                ..Default::default()
            }),
            expected
        );
    }

    #[rstest]
    #[case::no_team(false, None, OrganizationAction::Skip)]
    #[case::read_team(true, None, OrganizationAction::ReadTeamId)]
    #[case::no_team_id(true, Some(false), OrganizationAction::Skip)]
    #[case::team_selected(true, Some(true), OrganizationAction::ReadOrganization)]
    fn organization_fallback(
        #[case] team_present: bool,
        #[case] team_id: Option<bool>,
        #[case] expected: OrganizationAction,
    ) {
        assert_eq!(
            organization(&OrganizationFacts {
                token_present: true,
                client_present: true,
                token_id: Some(false),
                team_present,
                team_id,
                row_present: None,
                lookup_failed: false
            }),
            expected
        );
    }

    #[rstest]
    #[case::load_membership(false, false, false, false, false, MemberAction::ReadMembership)]
    #[case::read_limits(true, false, false, false, false, MemberAction::ReadLimits)]
    #[case::own_maximum(true, true, true, true, false, MemberAction::UseLimits)]
    #[case::default(true, true, false, true, false, MemberAction::ReadDefault)]
    #[case::invalid_default(true, true, false, false, false, MemberAction::UseLimits)]
    #[case::loaded_default(true, true, false, true, true, MemberAction::UseLimits)]
    fn member_stages(
        #[case] membership_loaded: bool,
        #[case] limits_read: bool,
        #[case] maximum_present: bool,
        #[case] default_id_string: bool,
        #[case] default_read: bool,
        #[case] expected: MemberAction,
    ) {
        assert_eq!(
            member(&MemberFacts {
                team_present: true,
                token_present: true,
                team_id: Some(true),
                user_id: Some(true),
                membership_loaded,
                limits_read,
                maximum_present,
                default_id_string,
                default_read
            }),
            expected
        );
    }

    #[rstest]
    #[case::comma_list(EmailFacts::Text(" a, ,b,a ".into()), vec!["a", "b", "a"])]
    #[case::list_keeps_space(EmailFacts::List(vec![Some(" a ".into()), None, Some("\u{1c}".into())]), vec![" a "])]
    #[case::python_whitespace(EmailFacts::Text("\u{1c}a\u{1f},\u{85}b\u{a0}".into()), vec!["a", "b"])]
    #[case::other(EmailFacts::Other, vec![])]
    fn email_normalization(#[case] facts: EmailFacts, #[case] expected: Vec<&str>) {
        assert_eq!(emails(&facts), expected);
    }
}
