use crate::{
    AuthHost, AuthorizeDenial, BudgetExceeded, Error,
    authorize::{AuthorizeRequest, Membership},
    host::MembershipRow,
};

/// `BudgetLimitEntry`: one window of a key's or team's `budget_limits`.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetWindow {
    pub duration: String,
    pub max_budget: f64,
}

/// The key's budget fields the authorization stage reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct KeyBudgetFacts<'a> {
    pub token: &'a str,
    pub org_id: Option<&'a str>,
    pub windows: &'a [BudgetWindow],
}

/// The team's budget fields the authorization stage reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct TeamBudgetFacts<'a> {
    pub team_id: &'a str,
    pub max_budget: Option<f64>,
    /// The spend recorded on the team row, which the live counter falls back to.
    pub spend: Option<f64>,
    pub soft_budget: Option<f64>,
    pub organization_id: Option<&'a str>,
    pub windows: &'a [BudgetWindow],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UserBudgetFacts<'a> {
    pub user_id: &'a str,
    pub max_budget: Option<f64>,
    pub spend: Option<f64>,
}

/// An end user that carries a budget row.
#[derive(Clone, Copy, Debug, Default)]
pub struct EndUserBudgetFacts<'a> {
    pub end_user_id: &'a str,
    pub max_budget: Option<f64>,
    pub spend: Option<f64>,
}

/// Everything the budget checks of the authorization stage decide on, read once by the host.
#[derive(Clone, Copy, Debug, Default)]
pub struct BudgetFacts<'a> {
    pub key: Option<KeyBudgetFacts<'a>>,
    pub team: Option<TeamBudgetFacts<'a>>,
    pub user: Option<UserBudgetFacts<'a>>,
    pub end_user: Option<EndUserBudgetFacts<'a>>,
    /// `general_settings.apply_user_budget_to_team_keys is True`
    pub apply_user_budget_to_team_keys: bool,
    /// `RouteChecks.is_info_route(route)`
    pub is_info_route: bool,
    /// `fail_closed_budget_enforcement_enabled()`
    pub fail_closed_budget_enforcement: bool,
    /// When a temporary budget increase still counts.
    pub now: f64,
}

/// The live spend counter a budget is compared against. The host owns the keyspace and falls back
/// to `recorded`, the spend on the row, when the counter is cold.
#[derive(Clone, Copy, Debug, PartialEq, strum::IntoStaticStr)]
pub enum SpendCounter<'a> {
    #[strum(serialize = "team")]
    Team { team_id: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "team_window")]
    TeamWindow { team_id: &'a str, window: &'a BudgetWindow },
    #[strum(serialize = "key_window")]
    KeyWindow { token: &'a str, window: &'a BudgetWindow },
    #[strum(serialize = "key")]
    Key { token: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "organization")]
    Organization { org_id: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "tag")]
    Tag { tag: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "model_access_group")]
    ModelAccessGroup { group: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "user")]
    User { user_id: &'a str, recorded: f64, max_budget: f64 },
    #[strum(serialize = "team_member")]
    TeamMember {
        user_id: &'a str,
        team_id: &'a str,
        recorded: f64,
        max_budget: f64,
    },
    #[strum(serialize = "end_user")]
    EndUser { end_user_id: &'a str, recorded: f64, max_budget: f64 },
}

/// A budget alert the host sends through its alerting. The core decides when one is due, the host
/// what it carries and who receives it.
#[derive(Clone, Copy, Debug, PartialEq, strum::IntoStaticStr)]
pub enum BudgetAlert<'a> {
    /// The team went over its budget.
    #[strum(serialize = "team")]
    Team { spend: f64, max_budget: f64 },
    /// The team's recorded spend crossed its soft budget.
    #[strum(serialize = "team_soft_budget")]
    TeamSoftBudget,
    /// The organization went over its budget.
    #[strum(serialize = "organization")]
    Organization { org_id: &'a str, spend: f64, max_budget: f64 },
    /// The user's spend against their budget, sent on every budgeted request.
    #[strum(serialize = "user")]
    User { spend: f64, max_budget: f64 },
    /// The team member's spend against their budget, which the host compares with the team's thresholds.
    #[strum(serialize = "team_member")]
    TeamMember { spend: f64, max_budget: f64 },
}

/// A ceiling read from a budget row, next to the spend recorded with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordedBudget {
    pub max_budget: f64,
    pub spend: f64,
}

/// A named entity's budget, such as a tag or a model access group.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedBudget {
    pub name: String,
    pub max_budget: f64,
    pub spend: f64,
}

/// The team member's budget: the membership's own row, and the team default it falls back to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemberBudget {
    pub spend: f64,
    pub budget: Option<BudgetRow>,
    /// The `max_budget` of the team's `team_member_budget_id` row, read only when the member has no
    /// budget of their own.
    pub team_default: Option<f64>,
}

/// The team, key, organization, tag, access group, user, team member and end user budgets, in
/// `check_budgets_for_request`'s order. Every scope runs, so every alert it is due is sent, and the
/// first scope that refused the request decides the error.
pub(crate) async fn check<H: AuthHost>(
    host: &H,
    request: &AuthorizeRequest<'_>,
    membership: Membership<'_, MembershipRow<H>>,
    matched_access_groups: &[String],
) -> Result<(), Error> {
    let facts = &request.budgets;
    let verdicts = [
        team_budget(host, facts).await,
        team_windows(host, facts).await,
        key_windows(host, facts).await,
        team_soft_budget(host, facts).await,
        organization_budget(host, facts).await,
        tag_budgets(host).await,
        model_access_group_budgets(host, matched_access_groups).await,
        user_budget(host, facts).await,
        team_member_budget(host, request, membership).await,
        end_user_budget(host, facts).await,
    ];
    verdicts.into_iter().collect()
}

async fn team_budget<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let Some(team) = facts.team else { return Ok(()) };
    let Some(max_budget) = team.max_budget else { return Ok(()) };
    let counter = SpendCounter::Team {
        team_id: team.team_id,
        recorded: team.spend.unwrap_or(0.0),
        max_budget,
    };
    let spend = host.spend(counter).await?;
    let verdict = self::team(team.team_id, spend, max_budget);
    if verdict.is_err() && facts.key.is_some() {
        host.budget_alert(BudgetAlert::Team { spend, max_budget }).await?;
    }
    Ok(verdict?)
}

async fn team_windows<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let Some(team) = facts.team else { return Ok(()) };
    for window in team.windows {
        let spend = host.spend(SpendCounter::TeamWindow { team_id: team.team_id, window }).await?;
        team_window(team.team_id, &window.duration, spend, window.max_budget)?;
    }
    Ok(())
}

async fn key_windows<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let Some(key) = facts.key else { return Ok(()) };
    for window in key.windows {
        let spend = host.spend(SpendCounter::KeyWindow { token: key.token, window }).await?;
        key_window(key.token, &window.duration, spend, window.max_budget)?;
    }
    Ok(())
}

async fn team_soft_budget<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let crossed = facts.key.is_some()
        && facts
            .team
            .is_some_and(|team| matches!((team.spend, team.soft_budget), (Some(spend), Some(soft)) if spend >= soft));
    if crossed {
        host.budget_alert(BudgetAlert::TeamSoftBudget).await?;
    }
    Ok(())
}

async fn organization_budget<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let Some(key) = facts.key else { return Ok(()) };
    let Some(org_id) = key.org_id.or(facts.team.and_then(|team| team.organization_id)) else {
        return Ok(());
    };
    let Some(budget) = host.organization_budget(org_id).await? else {
        return Ok(());
    };
    let counter = SpendCounter::Organization {
        org_id,
        recorded: budget.spend,
        max_budget: budget.max_budget,
    };
    let spend = host.spend(counter).await?;
    let verdict = organization(org_id, spend, budget.max_budget);
    if verdict.is_err() {
        let alert = BudgetAlert::Organization {
            org_id,
            spend,
            max_budget: budget.max_budget,
        };
        host.budget_alert(alert).await?;
    }
    Ok(verdict?)
}

async fn tag_budgets<H: AuthHost>(host: &H) -> Result<(), Error> {
    for budget in host.tag_budgets().await? {
        let counter = SpendCounter::Tag {
            tag: &budget.name,
            recorded: budget.spend,
            max_budget: budget.max_budget,
        };
        let spend = host.spend(counter).await?;
        tag(&budget.name, spend, budget.max_budget)?;
    }
    Ok(())
}

async fn model_access_group_budgets<H: AuthHost>(host: &H, groups: &[String]) -> Result<(), Error> {
    if groups.is_empty() {
        return Ok(());
    }
    let budgets = host.model_access_group_budgets(groups).await?;
    for budget in budgets.iter().filter(|budget| budget.max_budget > 0.0) {
        let counter = SpendCounter::ModelAccessGroup {
            group: &budget.name,
            recorded: budget.spend,
            max_budget: budget.max_budget,
        };
        let spend = host.spend(counter).await?;
        model_access_group(&budget.name, spend, budget.max_budget)?;
    }
    Ok(())
}

async fn user_budget<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    let Some(user) = facts.user else { return Ok(()) };
    let Some(max_budget) = user.max_budget else { return Ok(()) };
    if !user_budget_applies(facts.team.is_some(), facts.apply_user_budget_to_team_keys) {
        return Ok(());
    }
    let counter = SpendCounter::User {
        user_id: user.user_id,
        recorded: user.spend.unwrap_or(0.0),
        max_budget,
    };
    let spend = host.spend(counter).await?;
    host.budget_alert(BudgetAlert::User { spend, max_budget }).await?;
    Ok(self::user(user.user_id, spend, max_budget)?)
}

async fn team_member_budget<H: AuthHost>(
    host: &H,
    request: &AuthorizeRequest<'_>,
    membership: Membership<'_, MembershipRow<H>>,
) -> Result<(), Error> {
    let (Some(team), Some(user_id)) = (request.budgets.team, request.user_id) else {
        return Ok(());
    };
    let Membership::Read(row) = membership else {
        return Ok(());
    };
    let member = host.team_member_budget_state(row).await?;
    let Some(max_budget) = team_member_ceiling(member.budget.as_ref(), member.team_default, request.budgets.now) else {
        return Ok(());
    };
    if row.is_none() && request.budgets.fail_closed_budget_enforcement {
        return Err(AuthorizeDenial::TeamMemberWithoutMembership {
            user_id: user_id.to_owned(),
            team_id: team.team_id.to_owned(),
        }
        .into());
    }
    let counter = SpendCounter::TeamMember {
        user_id,
        team_id: team.team_id,
        recorded: member.spend,
        max_budget,
    };
    let spend = host.spend(counter).await?;
    if !max_budget.is_finite() {
        return Ok(());
    }
    host.budget_alert(BudgetAlert::TeamMember { spend, max_budget }).await?;
    Ok(team_member(user_id, team.team_id, spend, max_budget)?)
}

async fn end_user_budget<H: AuthHost>(host: &H, facts: &BudgetFacts<'_>) -> Result<(), Error> {
    if facts.is_info_route {
        return Ok(());
    }
    let Some(end_user) = facts.end_user else { return Ok(()) };
    let Some(max_budget) = end_user.max_budget else { return Ok(()) };
    let counter = SpendCounter::EndUser {
        end_user_id: end_user.end_user_id,
        recorded: end_user.spend.unwrap_or(0.0),
        max_budget,
    };
    let spend = host.spend(counter).await?;
    Ok(self::end_user(end_user.end_user_id, spend, max_budget)?)
}

/// A budget row's ceiling and its temporary increase, as `LiteLLM_BudgetTable` stores them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetRow {
    pub max_budget: Option<f64>,
    pub temp_budget_increase: Option<f64>,
    pub temp_budget_expiry: Option<f64>,
}

impl BudgetRow {
    fn active_temp_budget_increase(&self, now: f64) -> f64 {
        match (self.temp_budget_increase, self.temp_budget_expiry) {
            (Some(increase), Some(expiry)) if expiry > now => increase,
            _ => 0.0,
        }
    }

    fn effective_max_budget(&self, now: f64) -> Option<f64> {
        self.max_budget.map(|max_budget| max_budget + self.active_temp_budget_increase(now))
    }
}

/// The key a budget denial names.
#[derive(Clone, Copy, Debug)]
pub struct BudgetKey<'a> {
    pub token: &'a str,
    pub alias: Option<&'a str>,
    pub name: Option<&'a str>,
}

/// `resolve_effective_team_member_budget`
pub fn team_member_ceiling(member: Option<&BudgetRow>, team_default: Option<f64>, now: f64) -> Option<f64> {
    if let Some(ceiling) = member.and_then(|row| row.effective_max_budget(now)) {
        return Some(ceiling);
    }
    let default = team_default.filter(|max_budget| *max_budget > 0.0)?;
    Some(default + member.map_or(0.0, |row| row.active_temp_budget_increase(now)))
}

/// `virtual_key_max_budget_check`
pub fn key(key: BudgetKey<'_>, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!(
            "Budget has been exceeded! Key={} Current cost: {spend}, Max budget: {max_budget}",
            key_descriptor(key)
        ),
        "key",
        key.token,
    ))
}

/// `_virtual_key_multi_budget_check`, for one window.
pub fn key_window(token: &str, duration: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("ExceededBudget: Key over {duration} budget. Spend=${spend:.4}, Limit=${max_budget:.2}"),
        "key",
        token,
    ))
}

/// `team_max_budget_check`
pub fn team(team_id: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend <= max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("Budget has been exceeded! Team={team_id} Current cost: {spend}, Max budget: {max_budget}"),
        "team",
        team_id,
    ))
}

/// `_team_multi_budget_check`, for one window.
pub fn team_window(team_id: &str, duration: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!(
            "ExceededBudget: Team={team_id} over {duration} budget. Spend=${spend:.4}, Limit=${max_budget:.2}"
        ),
        "team",
        team_id,
    ))
}

/// `organization_max_budget_check`
pub fn organization(org_id: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("Budget has been exceeded! Organization={org_id} Current cost: {spend}, Max budget: {max_budget}"),
        "organization",
        org_id,
    ))
}

/// `tag_max_budget_check_for_tags`, for one tag.
pub fn tag(tag: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if spend <= max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("Budget has been exceeded! Tag={tag} Current cost: {spend}, Max budget: {max_budget}"),
        "tag",
        tag,
    ))
}

/// `_model_access_group_max_budget_check`, for one group. A non-positive budget means no budget.
pub fn model_access_group(group: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if max_budget <= 0.0 || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("Budget has been exceeded! Model access group={group} Current cost: {spend}, Max budget: {max_budget}"),
        "model_access_group",
        group,
    ))
}

/// Whether the user's own budget applies to this request. A team key spends the team's budget
/// unless `apply_user_budget_to_team_keys` is set.
pub fn user_budget_applies(team_key: bool, apply_user_budget_to_team_keys: bool) -> bool {
    !team_key || apply_user_budget_to_team_keys
}

/// `_user_max_budget_check`
pub fn user(user_id: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("ExceededBudget: User={user_id} over budget. Spend={spend}, Budget={max_budget}"),
        "user",
        user_id,
    ))
}

/// `_check_team_member_budget`
pub fn team_member(user_id: &str, team_id: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!(
            "Budget has been exceeded! User={user_id} in Team={team_id} Current cost: {spend}, Max budget: {max_budget}"
        ),
        "team_member",
        &format!("{user_id}:{team_id}"),
    ))
}

/// `check_end_user_budget`
pub fn end_user(end_user_id: &str, spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
    if spend <= max_budget {
        return Ok(());
    }
    Err(exceeded(
        spend,
        max_budget,
        format!("ExceededBudget: End User={end_user_id} over budget. Spend={spend}, Budget={max_budget}"),
        "end_user",
        end_user_id,
    ))
}

fn exceeded(spend: f64, max_budget: f64, message: String, entity_type: &str, entity_id: &str) -> BudgetExceeded {
    BudgetExceeded {
        current_cost: spend,
        max_budget,
        message,
        entity_type: Some(entity_type.to_owned()),
        entity_id: Some(entity_id.to_owned()),
    }
}

/// The key's alias, followed by its masked name when it has one, so a denial never shows the secret.
fn key_descriptor(key: BudgetKey<'_>) -> String {
    let label = key.alias.unwrap_or("key");
    match key.name.filter(|name| is_masked_key_name(name)) {
        Some(name) => format!("{label} ({name})"),
        None => label.to_owned(),
    }
}

/// `_MASKED_KEY_NAME_RE`: `sk-...` alone or followed by exactly four visible characters.
fn is_masked_key_name(name: &str) -> bool {
    let Some(tail) = name.strip_prefix("sk-...") else {
        return false;
    };
    tail.is_empty()
        || tail.chars().count() == 4 && tail.chars().all(|c| !c.is_whitespace() && !is_c0_or_c1_control(c))
}

fn is_c0_or_c1_control(c: char) -> bool {
    matches!(c, '\u{0}'..='\u{1f}' | '\u{7f}'..='\u{9f}')
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const NOW: f64 = 1_000.0;

    fn row(max_budget: Option<f64>, increase: Option<f64>, expiry: Option<f64>) -> BudgetRow {
        BudgetRow {
            max_budget,
            temp_budget_increase: increase,
            temp_budget_expiry: expiry,
        }
    }

    #[rstest]
    #[case::member_budget_wins(Some(row(Some(10.0), None, None)), Some(50.0), Some(10.0))]
    #[case::member_increase_active(Some(row(Some(10.0), Some(5.0), Some(NOW + 1.0))), None, Some(15.0))]
    #[case::member_increase_expired_at_now(Some(row(Some(10.0), Some(5.0), Some(NOW))), None, Some(10.0))]
    #[case::increase_without_expiry(Some(row(Some(10.0), Some(5.0), None)), None, Some(10.0))]
    #[case::default_when_member_unbudgeted(Some(row(None, None, None)), Some(50.0), Some(50.0))]
    #[case::default_plus_member_increase(Some(row(None, Some(5.0), Some(NOW + 1.0))), Some(50.0), Some(55.0))]
    #[case::default_without_member(None, Some(50.0), Some(50.0))]
    #[case::zero_default_is_none(None, Some(0.0), None)]
    #[case::negative_default_is_none(None, Some(-1.0), None)]
    #[case::nothing(None, None, None)]
    fn team_member_ceiling_resolves_like_python(
        #[case] member: Option<BudgetRow>,
        #[case] team_default: Option<f64>,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(team_member_ceiling(member.as_ref(), team_default, NOW), expected);
    }

    type Check = fn(f64, f64) -> Result<(), BudgetExceeded>;

    fn key_check(spend: f64, max_budget: f64) -> Result<(), BudgetExceeded> {
        key(
            BudgetKey {
                token: "hashed",
                alias: None,
                name: None,
            },
            spend,
            max_budget,
        )
    }

    #[rstest]
    #[case::key(key_check as Check, true, false)]
    #[case::key_window((|s, m| key_window("hashed", "1d", s, m)) as Check, true, false)]
    #[case::team((|s, m| team("t", s, m)) as Check, false, false)]
    #[case::team_window((|s, m| team_window("t", "1d", s, m)) as Check, true, false)]
    #[case::organization((|s, m| organization("o", s, m)) as Check, true, false)]
    #[case::tag((|s, m| tag("prod", s, m)) as Check, false, false)]
    #[case::model_access_group((|s, m| model_access_group("g", s, m)) as Check, true, true)]
    #[case::user((|s, m| user("u", s, m)) as Check, true, false)]
    #[case::team_member((|s, m| team_member("u", "t", s, m)) as Check, true, false)]
    #[case::end_user((|s, m| end_user("e", s, m)) as Check, false, false)]
    fn each_budget_draws_its_own_line(
        #[case] check: Check,
        #[case] denies_at_ceiling: bool,
        #[case] denies_infinite_budget: bool,
    ) {
        assert!(check(9.0, 10.0).is_ok());
        assert_eq!(check(10.0, 10.0).is_err(), denies_at_ceiling);
        assert!(check(11.0, 10.0).is_err());
        assert_eq!(check(f64::INFINITY, f64::INFINITY).is_err(), denies_infinite_budget);
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::negative(-5.0)]
    fn model_access_group_without_positive_budget_never_denies(#[case] max_budget: f64) {
        assert_eq!(model_access_group("g", 100.0, max_budget), Ok(()));
    }

    #[rstest]
    #[case::personal_key(false, false, true)]
    #[case::team_key(true, false, false)]
    #[case::team_key_opted_in(true, true, true)]
    fn user_budget_skips_team_keys_unless_opted_in(
        #[case] team_key: bool,
        #[case] apply_to_team_keys: bool,
        #[case] applies: bool,
    ) {
        assert_eq!(user_budget_applies(team_key, apply_to_team_keys), applies);
    }

    #[rstest]
    #[case::no_alias(None, None, "Key=key Current")]
    #[case::alias(Some("prod"), None, "Key=prod Current")]
    #[case::masked_name(Some("prod"), Some("sk-...abcd"), "Key=prod (sk-...abcd) Current")]
    #[case::masked_name_without_tail(None, Some("sk-..."), "Key=key (sk-...) Current")]
    #[case::raw_secret_hidden(Some("prod"), Some("sk-abcdefgh"), "Key=prod Current")]
    #[case::long_tail_hidden(Some("prod"), Some("sk-...abcde"), "Key=prod Current")]
    #[case::whitespace_tail_hidden(Some("prod"), Some("sk-...ab d"), "Key=prod Current")]
    #[case::control_tail_hidden(Some("prod"), Some("sk-...ab\u{85}d"), "Key=prod Current")]
    fn key_denial_names_alias_and_only_a_masked_name(
        #[case] alias: Option<&str>,
        #[case] name: Option<&str>,
        #[case] expected: &str,
    ) {
        let denial = key(
            BudgetKey {
                token: "hashed",
                alias,
                name,
            },
            10.0,
            10.0,
        )
        .unwrap_err();
        assert!(denial.message.contains(expected), "{}", denial.message);
        assert_eq!(denial.entity_id.as_deref(), Some("hashed"));
    }

    #[test]
    fn denial_carries_spend_ceiling_and_entity() {
        assert_eq!(
            team_member("u", "t", 12.5, 10.0),
            Err(BudgetExceeded {
                current_cost: 12.5,
                max_budget: 10.0,
                message: "Budget has been exceeded! User=u in Team=t Current cost: 12.5, Max budget: 10".to_owned(),
                entity_type: Some("team_member".to_owned()),
                entity_id: Some("u:t".to_owned()),
            })
        );
    }

    use std::collections::HashMap;

    use crate::testing::ScriptedHost;

    const KEY: KeyBudgetFacts<'static> = KeyBudgetFacts {
        token: "k1",
        org_id: None,
        windows: &[],
    };
    const TEAM: TeamBudgetFacts<'static> = TeamBudgetFacts {
        team_id: "t1",
        max_budget: Some(10.0),
        spend: Some(1.0),
        soft_budget: None,
        organization_id: None,
        windows: &[],
    };
    const USER: UserBudgetFacts<'static> = UserBudgetFacts {
        user_id: "u1",
        max_budget: Some(5.0),
        spend: Some(1.0),
    };

    fn authorize_request(budgets: BudgetFacts<'static>) -> AuthorizeRequest<'static> {
        AuthorizeRequest {
            user_id: Some("u1"),
            budgets,
            ..AuthorizeRequest::default()
        }
    }

    fn host(spends: &[(&'static str, f64)]) -> ScriptedHost {
        ScriptedHost {
            spends: spends.iter().copied().collect::<HashMap<_, _>>(),
            ..ScriptedHost::default()
        }
    }

    async fn run(host: &ScriptedHost, budgets: BudgetFacts<'static>, membership: Membership<'_, String>) -> Result<(), Error> {
        check(host, &authorize_request(budgets), membership, &[]).await
    }

    #[tokio::test]
    async fn every_scope_runs_and_the_first_refusal_decides() {
        let host = host(&[("team t1", 11.0), ("user u1", 6.0)]);
        let budgets = BudgetFacts {
            key: Some(KEY),
            team: Some(TEAM),
            user: Some(USER),
            apply_user_budget_to_team_keys: true,
            ..BudgetFacts::default()
        };
        let denial = run(&host, budgets, Membership::Unread).await.unwrap_err();
        assert!(matches!(denial, Error::BudgetExceeded(ref exceeded) if exceeded.entity_type.as_deref() == Some("team")));
        assert_eq!(
            host.calls(),
            [
                "spend team t1",
                "alert team",
                "tag_budgets",
                "spend user u1",
                "alert user",
            ]
        );
    }

    #[rstest]
    #[case::with_a_key(true, &["spend team t1", "alert team", "tag_budgets"])]
    #[case::without_a_key(false, &["spend team t1", "tag_budgets"])]
    #[tokio::test]
    async fn an_over_budget_team_alerts_only_for_a_key(#[case] with_key: bool, #[case] calls: &[&str]) {
        let host = host(&[("team t1", 10.5)]);
        let budgets = BudgetFacts {
            key: with_key.then_some(KEY),
            team: Some(TEAM),
            ..BudgetFacts::default()
        };
        assert!(run(&host, budgets, Membership::Unread).await.is_err());
        assert_eq!(host.calls(), calls);
    }

    #[rstest]
    #[case::personal_key(false, false, true)]
    #[case::team_key(true, false, false)]
    #[case::team_key_opted_in(true, true, true)]
    #[tokio::test]
    async fn the_user_budget_reads_spend_only_where_it_applies(
        #[case] team_key: bool,
        #[case] apply_to_team_keys: bool,
        #[case] read: bool,
    ) {
        let host = host(&[]);
        let budgets = BudgetFacts {
            team: team_key.then_some(TeamBudgetFacts { max_budget: None, ..TEAM }),
            user: Some(USER),
            apply_user_budget_to_team_keys: apply_to_team_keys,
            ..BudgetFacts::default()
        };
        assert_eq!(run(&host, budgets, Membership::Unread).await, Ok(()));
        assert_eq!(host.calls().contains(&"spend user u1".to_owned()), read);
    }

    #[rstest]
    #[case::from_the_key(Some("o1"), Some("o2"), Some("o1"))]
    #[case::from_the_team(None, Some("o2"), Some("o2"))]
    #[case::none(None, None, None)]
    #[tokio::test]
    async fn the_organization_is_the_keys_else_the_teams(
        #[case] key_org: Option<&'static str>,
        #[case] team_org: Option<&'static str>,
        #[case] read: Option<&str>,
    ) {
        let host = host(&[]);
        let budgets = BudgetFacts {
            key: Some(KeyBudgetFacts { org_id: key_org, ..KEY }),
            team: Some(TeamBudgetFacts {
                max_budget: None,
                organization_id: team_org,
                ..TEAM
            }),
            ..BudgetFacts::default()
        };
        assert_eq!(run(&host, budgets, Membership::Unread).await, Ok(()));
        assert_eq!(
            host.calls().first().cloned(),
            Some(read.map_or("tag_budgets".to_owned(), |org| format!("organization_budget {org}")))
        );
    }

    #[tokio::test]
    async fn an_organization_over_budget_alerts_and_refuses() {
        let host = ScriptedHost {
            organization_budgets: HashMap::from([("o1", RecordedBudget { max_budget: 10.0, spend: 2.0 })]),
            ..host(&[("organization o1", 10.0)])
        };
        let budgets = BudgetFacts {
            key: Some(KeyBudgetFacts { org_id: Some("o1"), ..KEY }),
            ..BudgetFacts::default()
        };
        let denial = run(&host, budgets, Membership::Unread).await.unwrap_err();
        assert!(matches!(denial, Error::BudgetExceeded(ref exceeded) if exceeded.entity_id.as_deref() == Some("o1")));
        assert_eq!(
            host.calls(),
            ["organization_budget o1", "spend organization o1", "alert organization", "tag_budgets"]
        );
    }

    #[tokio::test]
    async fn an_access_group_without_a_positive_budget_reads_no_spend() {
        let host = ScriptedHost {
            model_access_group_budgets: vec![
                NamedBudget { name: "free".to_owned(), max_budget: 0.0, spend: 0.0 },
                NamedBudget { name: "paid".to_owned(), max_budget: 5.0, spend: 0.0 },
            ],
            ..host(&[])
        };
        let groups = ["free".to_owned(), "paid".to_owned()];
        let result = check(&host, &authorize_request(BudgetFacts::default()), Membership::Unread, &groups).await;
        assert_eq!(result, Ok(()));
        assert_eq!(
            host.calls(),
            ["tag_budgets", "model_access_group_budgets free,paid", "spend model_access_group paid"]
        );
    }

    #[rstest]
    #[case::fail_closed(true, Err(AuthorizeDenial::TeamMemberWithoutMembership {
        user_id: "u1".to_owned(),
        team_id: "t1".to_owned(),
    }.into()))]
    #[case::fail_open(false, Ok(()))]
    #[tokio::test]
    async fn a_member_without_a_membership_row_is_refused_only_when_fail_closed(
        #[case] fail_closed: bool,
        #[case] expected: Result<(), Error>,
    ) {
        let host = ScriptedHost {
            member_budget: MemberBudget {
                team_default: Some(10.0),
                ..MemberBudget::default()
            },
            ..host(&[])
        };
        let budgets = BudgetFacts {
            team: Some(TeamBudgetFacts { max_budget: None, ..TEAM }),
            fail_closed_budget_enforcement: fail_closed,
            ..BudgetFacts::default()
        };
        assert_eq!(run(&host, budgets, Membership::Read(None)).await, expected);
        assert_eq!(host.calls().contains(&"spend team_member u1 t1".to_owned()), !fail_closed);
    }

    #[rstest]
    #[case::under(11.0, Ok(()), true)]
    #[case::at_the_ceiling(10.0, Err(()), true)]
    #[case::unlimited(f64::INFINITY, Ok(()), false)]
    #[tokio::test]
    async fn the_team_member_budget_alerts_before_it_refuses(
        #[case] ceiling: f64,
        #[case] expected: Result<(), ()>,
        #[case] alerted: bool,
    ) {
        let row = "membership".to_owned();
        let host = ScriptedHost {
            member_budget: MemberBudget {
                spend: 1.0,
                budget: Some(BudgetRow {
                    max_budget: Some(ceiling),
                    ..BudgetRow::default()
                }),
                team_default: None,
            },
            ..host(&[("team_member u1 t1", 10.0)])
        };
        let budgets = BudgetFacts {
            team: Some(TeamBudgetFacts { max_budget: None, ..TEAM }),
            ..BudgetFacts::default()
        };
        let result = run(&host, budgets, Membership::Read(Some(&row))).await;
        assert_eq!(result.map_err(|_| ()), expected);
        assert_eq!(host.calls().contains(&"alert team_member".to_owned()), alerted);
        assert!(host.calls().contains(&"spend team_member u1 t1".to_owned()));
    }

    #[rstest]
    #[case::info_route(true, false)]
    #[case::inference_route(false, true)]
    #[tokio::test]
    async fn the_end_user_budget_skips_info_routes(#[case] is_info_route: bool, #[case] read: bool) {
        let host = host(&[]);
        let budgets = BudgetFacts {
            end_user: Some(EndUserBudgetFacts {
                end_user_id: "e1",
                max_budget: Some(1.0),
                spend: None,
            }),
            is_info_route,
            ..BudgetFacts::default()
        };
        assert_eq!(run(&host, budgets, Membership::Unread).await, Ok(()));
        assert_eq!(host.calls().contains(&"spend end_user e1".to_owned()), read);
    }

    #[tokio::test]
    async fn a_team_soft_budget_crossing_alerts_without_refusing() {
        let host = host(&[]);
        let budgets = BudgetFacts {
            key: Some(KEY),
            team: Some(TeamBudgetFacts {
                max_budget: None,
                spend: Some(3.0),
                soft_budget: Some(3.0),
                ..TEAM
            }),
            ..BudgetFacts::default()
        };
        assert_eq!(run(&host, budgets, Membership::Unread).await, Ok(()));
        assert_eq!(host.calls(), ["alert team_soft_budget", "tag_budgets"]);
    }

    #[test]
    fn window_denial_rounds_spend_and_limit() {
        let denial = team_window("t", "30d", 10.123456, 10.0).unwrap_err();
        assert_eq!(
            denial.message,
            "ExceededBudget: Team=t over 30d budget. Spend=$10.1235, Limit=$10.00"
        );
    }
}
