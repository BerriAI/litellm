use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum ModelScope {
    Team,
    EndUser,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelCheckFacts {
    pub scope: ModelScope,
    pub entity_id: Option<String>,
    pub budget_count: usize,
    pub model_count: usize,
}

pub fn model_checks(facts: &ModelCheckFacts) -> Vec<usize> {
    if facts.budget_count == 0
        || facts.entity_id.is_none()
        || (facts.scope == ModelScope::EndUser && facts.entity_id.as_deref() == Some(""))
    {
        return vec![];
    }
    (0..facts.model_count).collect()
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct BudgetFields {
    pub max_budget: Option<f64>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub model_budget_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::VariantArray)]
pub enum EndUserMode {
    Models,
    Token,
    Default,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EndUserFacts {
    pub mode: EndUserMode,
    pub object_present: bool,
    pub end_user_id: Option<String>,
    pub prisma_present: bool,
    pub key_budget_id: Option<String>,
    pub key_read: bool,
    pub key_budget: Option<BudgetFields>,
    pub global_config_read: bool,
    pub global_budget_id: Option<String>,
    pub global_read: bool,
    pub global_budget: Option<BudgetFields>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum EndUserSource {
    Own,
    ReadKey,
    Key,
    ReadGlobalConfig,
    ReadGlobal,
    Global,
    None,
}

pub fn end_user_source(facts: &EndUserFacts) -> EndUserSource {
    use EndUserSource::*;
    if facts.mode != EndUserMode::Default && facts.object_present {
        return Own;
    }
    if facts.mode == EndUserMode::Models
        && (!facts.prisma_present || facts.end_user_id.as_deref().is_none_or(str::is_empty))
    {
        return None;
    }
    if facts.key_budget_id.is_some() {
        if !facts.key_read {
            return ReadKey;
        }
        if facts.key_budget.is_some() {
            return Key;
        }
    }
    if !facts.global_config_read {
        return ReadGlobalConfig;
    }
    if facts.global_budget_id.is_none() {
        return None;
    }
    if !facts.global_read {
        return ReadGlobal;
    }
    if facts.global_budget.is_some() {
        Global
    } else {
        None
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TokenDefaultsFacts {
    pub current: BudgetFields,
    pub keep_token_limits: bool,
    pub default_present: bool,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum TokenField {
    MaxBudget,
    Tpm,
    Rpm,
    Tpd,
    Models,
}

pub fn token_assignments(facts: &TokenDefaultsFacts) -> Vec<TokenField> {
    if !facts.default_present {
        return vec![];
    }
    [
        (TokenField::MaxBudget, facts.current.max_budget.is_none()),
        (TokenField::Tpm, facts.current.tpm_limit.is_none()),
        (TokenField::Rpm, facts.current.rpm_limit.is_none()),
        (TokenField::Tpd, facts.current.tpd_limit.is_none()),
        (
            TokenField::Models,
            facts.current.model_budget_count.is_none(),
        ),
    ]
    .into_iter()
    .filter_map(|(field, missing)| (!facts.keep_token_limits || missing).then_some(field))
    .collect()
}

#[derive(Clone, Debug, Deserialize)]
pub struct MembershipSnapshot {
    pub budget_id: Option<String>,
    pub model_budget_count: Option<usize>,
}

#[derive(Clone, Copy, Debug, Deserialize, strum::VariantArray)]
pub enum LookupFailure {
    Membership,
    Default,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MemberFacts {
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub prisma_present: bool,
    pub default_id: Option<String>,
    pub skip_budget_checks: bool,
    pub membership_read: bool,
    pub membership: Option<MembershipSnapshot>,
    pub default_read: bool,
    pub default_budget: Option<BudgetFields>,
    pub lookup_failure: Option<LookupFailure>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum MemberSource {
    Done,
    ReadMembership,
    ReadDefault,
    Member,
    Default,
    Clear,
    RaiseLookup,
    SkipLookup,
}

pub fn member_source(facts: &MemberFacts) -> MemberSource {
    use MemberSource::*;
    if facts.user_id.is_none() || facts.team_id.is_none() || !facts.prisma_present {
        return Done;
    }
    if facts.lookup_failure.is_some() {
        return if facts.skip_budget_checks {
            SkipLookup
        } else {
            RaiseLookup
        };
    }
    if !facts.membership_read {
        return ReadMembership;
    }
    if facts.membership.as_ref().is_some_and(|member| {
        member.budget_id.is_some()
            && member.budget_id != facts.default_id
            && member.model_budget_count.is_some_and(|count| count > 0)
    }) {
        return Member;
    }
    if facts.default_id.is_none() {
        return Done;
    }
    if !facts.default_read {
        return ReadDefault;
    }
    if facts
        .default_budget
        .as_ref()
        .is_some_and(|budget| budget.model_budget_count.is_some())
    {
        Default
    } else {
        Clear
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct IncreaseFacts {
    pub number: Option<f64>,
    pub truthy: bool,
    pub type_name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TemporaryFacts {
    pub max_budget: Option<f64>,
    pub metadata_read: bool,
    pub has_increase: bool,
    pub has_expiry: bool,
    pub expires_at: Option<i64>,
    pub now: Option<i64>,
    pub increase: Option<IncreaseFacts>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, strum::VariantArray)]
pub enum TemporaryRead {
    Metadata,
    Expiry,
    Now,
    Increase,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct TemporaryDecision {
    pub read: Option<TemporaryRead>,
    pub copy: bool,
    pub max_budget: Option<f64>,
    pub invalid_increase_type: Option<String>,
}

fn temp_read(read: TemporaryRead) -> TemporaryDecision {
    TemporaryDecision {
        read: Some(read),
        ..TemporaryDecision::default()
    }
}

pub fn temporary(facts: &TemporaryFacts) -> TemporaryDecision {
    let Some(limit) = facts.max_budget else {
        return TemporaryDecision::default();
    };
    if !facts.metadata_read {
        return temp_read(TemporaryRead::Metadata);
    }
    if !facts.has_increase || !facts.has_expiry {
        return TemporaryDecision::default();
    }
    let Some(expiry) = facts.expires_at else {
        return temp_read(TemporaryRead::Expiry);
    };
    let Some(now) = facts.now else {
        return temp_read(TemporaryRead::Now);
    };
    if expiry <= now {
        return TemporaryDecision::default();
    }
    let Some(increase) = &facts.increase else {
        return temp_read(TemporaryRead::Increase);
    };
    if !increase.truthy {
        return TemporaryDecision::default();
    }
    match increase.number {
        Some(amount) => TemporaryDecision {
            copy: true,
            max_budget: Some(limit + amount),
            ..TemporaryDecision::default()
        },
        None => TemporaryDecision {
            invalid_increase_type: Some(increase.type_name.clone()),
            ..TemporaryDecision::default()
        },
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct NumericFacts {
    pub number: Option<f64>,
    pub boolean: bool,
}

pub fn percentage(facts: &NumericFacts) -> Option<f64> {
    if facts.boolean {
        return None;
    }
    facts.number.filter(|pct| *pct > 0.0 && *pct <= 1.0)
}

#[derive(Clone, Debug, Deserialize)]
pub struct ThrottleFacts {
    pub opt_in: bool,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub percentage: Option<NumericFacts>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct ThrottleDecision {
    pub read_percentage: bool,
    pub percentage: Option<f64>,
}

pub fn throttle(facts: &ThrottleFacts) -> ThrottleDecision {
    if !facts.opt_in || (facts.tpm_limit.is_none() && facts.rpm_limit.is_none()) {
        return ThrottleDecision::default();
    }
    match &facts.percentage {
        None => ThrottleDecision {
            read_percentage: true,
            ..ThrottleDecision::default()
        },
        Some(raw) => ThrottleDecision {
            percentage: percentage(raw),
            ..ThrottleDecision::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::team_empty(ModelScope::Team, Some(""), vec![0, 1])]
    #[case::team_none(ModelScope::Team, None, vec![])]
    #[case::enduser_empty(ModelScope::EndUser, Some(""), vec![])]
    #[case::enduser(ModelScope::EndUser, Some("user"), vec![0, 1])]
    fn model_identity_gates(
        #[case] scope: ModelScope,
        #[case] identity: Option<&str>,
        #[case] expected: Vec<usize>,
    ) {
        assert_eq!(
            model_checks(&ModelCheckFacts {
                scope,
                entity_id: identity.map(str::to_owned),
                budget_count: 1,
                model_count: 2
            }),
            expected
        );
    }

    #[rstest]
    #[case::zero(0.0, false, None)]
    #[case::negative(-0.5, false, None)]
    #[case::nan(f64::NAN, false, None)]
    #[case::infinite(f64::INFINITY, false, None)]
    #[case::one(1.0, false, Some(1.0))]
    #[case::boolean(1.0, true, None)]
    fn numeric_percentages(
        #[case] number: f64,
        #[case] boolean: bool,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(
            percentage(&NumericFacts {
                number: Some(number),
                boolean
            }),
            expected
        );
    }

    #[rstest]
    #[case::expired(10, 10, 3.0, false, None)]
    #[case::before(9, 10, 3.0, false, None)]
    #[case::active(11, 10, 3.0, true, Some(13.0))]
    #[case::negative(11, 10, -3.0, true, Some(7.0))]
    #[case::zero(11, 10, 0.0, false, None)]
    fn temporary_copy_boundary(
        #[case] expiry: i64,
        #[case] now: i64,
        #[case] amount: f64,
        #[case] copy: bool,
        #[case] expected: Option<f64>,
    ) {
        let facts = TemporaryFacts {
            max_budget: Some(10.0),
            metadata_read: true,
            has_increase: true,
            has_expiry: true,
            expires_at: Some(expiry),
            now: Some(now),
            increase: Some(IncreaseFacts {
                number: Some(amount),
                truthy: amount != 0.0,
                type_name: "float".to_owned(),
            }),
        };
        let result = temporary(&facts);
        assert_eq!(
            (result.copy, result.max_budget, result.read),
            (copy, expected, None)
        );
    }

    #[rstest]
    #[case::member_override(Some("member"), Some(1), Some("default"), false, MemberSource::Member)]
    #[case::no_override(None, Some(1), None, false, MemberSource::Done)]
    #[case::empty_override(
        Some("member"),
        Some(0),
        Some("default"),
        false,
        MemberSource::ReadDefault
    )]
    #[case::current_default(
        Some("default"),
        Some(1),
        Some("default"),
        false,
        MemberSource::ReadDefault
    )]
    #[case::missing_default(None, None, Some("default"), true, MemberSource::Clear)]
    fn member_default_staging(
        #[case] member_id: Option<&str>,
        #[case] count: Option<usize>,
        #[case] default_id: Option<&str>,
        #[case] default_read: bool,
        #[case] expected: MemberSource,
    ) {
        let facts = MemberFacts {
            user_id: Some(String::new()),
            team_id: Some(String::new()),
            prisma_present: true,
            default_id: default_id.map(str::to_owned),
            skip_budget_checks: false,
            membership_read: true,
            membership: Some(MembershipSnapshot {
                budget_id: member_id.map(str::to_owned),
                model_budget_count: count,
            }),
            default_read,
            default_budget: None,
            lookup_failure: None,
        };
        assert_eq!(member_source(&facts), expected);
    }

    #[rstest]
    #[case::replace(false, true, vec![TokenField::MaxBudget, TokenField::Tpm, TokenField::Rpm, TokenField::Tpd, TokenField::Models])]
    #[case::keep_empty(true, true, vec![TokenField::Rpm, TokenField::Tpd])]
    #[case::no_default(false, false, vec![])]
    fn token_assignment_order(
        #[case] keep: bool,
        #[case] present: bool,
        #[case] expected: Vec<TokenField>,
    ) {
        let facts = TokenDefaultsFacts {
            current: BudgetFields {
                max_budget: Some(0.0),
                tpm_limit: Some(0),
                model_budget_count: Some(0),
                ..BudgetFields::default()
            },
            keep_token_limits: keep,
            default_present: present,
        };
        assert_eq!(token_assignments(&facts), expected);
    }
}
