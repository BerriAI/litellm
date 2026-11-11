use crate::{
    AuthHost, BudgetExceeded, Error,
    budgets::SpendCounter,
    host::{KeyRow, UserRow},
    key::{KeyRecord, KeySettings},
};

/// When a key's spend warns before it reaches its max budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxBudgetWarning {
    /// No alert emails are configured: warn from this share of the budget, until the budget is reached.
    Default { share: f64 },
    /// Alert emails are configured for these percentages: warn from the lowest one. `None` when no
    /// percentage is a number, so no warning is due.
    Emails { lowest_percent: Option<f64> },
}

impl Default for MaxBudgetWarning {
    fn default() -> Self {
        Self::Emails { lowest_percent: None }
    }
}

/// The key's own budget fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyBudgetRecord {
    pub max_budget: Option<f64>,
    /// The spend recorded on the key row.
    pub spend: Option<f64>,
    pub soft_budget: Option<f64>,
    pub key_alias: Option<String>,
    pub key_name: Option<String>,
    pub warning: MaxBudgetWarning,
    /// The share of its limits an over-budget key is throttled to instead of refused, when it opted in.
    pub throttle_share: Option<f64>,
}

/// A key path budget alert the host sends through its alerting, for the key and owner rows it
/// carries. The core decides when one is due.
#[derive(Clone, Copy, Debug, PartialEq, strum::IntoStaticStr)]
pub enum KeyBudgetAlert {
    /// The key's team member spend reached the member's budget.
    #[strum(serialize = "team_member")]
    TeamMember { spend: f64, max_budget: f64 },
    /// The proxy's global spend, sent on every request while the proxy has a budget.
    #[strum(serialize = "proxy")]
    Proxy { spend: f64 },
    /// The key's recorded spend nears its max budget.
    #[strum(serialize = "max_budget_warning")]
    MaxBudgetWarning,
    /// The key's live spend against its max budget, sent on every request while it has one.
    #[strum(serialize = "token_budget")]
    TokenBudget { spend: f64 },
    /// The key's recorded spend crossed its soft budget.
    #[strum(serialize = "soft_budget")]
    SoftBudget,
}

/// `_MASKED_KEY_NAME_RE`: `sk-...` with the last four characters of the key, or nothing.
fn is_masked_key_name(key_name: &str) -> bool {
    key_name.strip_prefix("sk-...").is_some_and(|tail| {
        tail.is_empty()
            || (tail.chars().count() == 4
                && tail
                    .chars()
                    .all(|c| !c.is_whitespace() && !matches!(c as u32, 0..=0x1f | 0x7f..=0x9f)))
    })
}

/// How the budget denial names the key: its alias, with its masked name when that still looks
/// masked, since the name has no enforced shape.
fn key_descriptor(budget: &KeyBudgetRecord) -> String {
    let label = budget.key_alias.as_deref().filter(|alias| !alias.is_empty()).unwrap_or("key");
    match budget.key_name.as_deref().filter(|name| is_masked_key_name(name)) {
        Some(name) => format!("{label} ({name})"),
        None => label.to_owned(),
    }
}

/// `virtual_key_max_budget_check`: the key's live spend against its max budget. A key that opted
/// into throttling is throttled instead of refused.
pub async fn key_max_budget<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
    key: &KeyRow<H>,
    owner: Option<&UserRow<H>>,
) -> Result<(), Error> {
    let budget = &record.budget;
    let (Some(max_budget), Some(token)) = (budget.max_budget, record.token.as_deref()) else {
        return Ok(());
    };
    let counter = SpendCounter::Key {
        token,
        recorded: budget.spend.unwrap_or(0.0),
        max_budget,
    };
    let spend = host.spend(counter).await?;
    host.key_budget_alert(key, owner, KeyBudgetAlert::TokenBudget { spend }).await?;
    if !max_budget.is_finite() || spend < max_budget {
        return Ok(());
    }
    if let Some(share) = budget.throttle_share {
        return host.throttle_key(key, share).await;
    }
    Err(BudgetExceeded {
        current_cost: spend,
        max_budget,
        message: format!(
            "Budget has been exceeded! Key={} Current cost: {spend}, Max budget: {max_budget}",
            key_descriptor(budget)
        ),
        entity_type: Some("key".to_owned()),
        entity_id: Some(token.to_owned()),
    }
    .into())
}

/// `virtual_key_max_budget_alert_check`: warns before the recorded spend reaches the max budget.
pub fn max_budget_warning_due(budget: &KeyBudgetRecord) -> bool {
    let (Some(max_budget), Some(spend)) = (budget.max_budget, budget.spend.filter(|spend| *spend > 0.0)) else {
        return false;
    };
    match budget.warning {
        MaxBudgetWarning::Emails { lowest_percent } => {
            lowest_percent.is_some_and(|percent| spend >= max_budget * (percent / 100.0))
        }
        MaxBudgetWarning::Default { share } => spend >= max_budget * share && spend < max_budget,
    }
}

/// `virtual_key_soft_budget_check`: the recorded spend crossed a set soft budget.
pub fn soft_budget_due(budget: &KeyBudgetRecord) -> bool {
    matches!(
        (budget.soft_budget.filter(|soft| *soft != 0.0), budget.spend),
        (Some(soft_budget), Some(spend)) if spend >= soft_budget
    )
}

/// The key's own budgets, in `BudgetChecks.key`'s order: the warning, the max budget on LLM routes,
/// the soft budget, then the per-model budgets the host still checks. A free request skips them all.
pub(crate) async fn key_budgets<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
    key: &KeyRow<H>,
    owner: Option<&UserRow<H>>,
    is_llm_api_route: bool,
) -> Result<(), Error> {
    if host.key_skips_budget_checks(key).await? {
        return Ok(());
    }
    let budget = &record.budget;
    if max_budget_warning_due(budget) {
        host.key_budget_alert(key, owner, KeyBudgetAlert::MaxBudgetWarning).await?;
    }
    if is_llm_api_route {
        key_max_budget(host, record, key, owner).await?;
    }
    if soft_budget_due(budget) {
        host.key_budget_alert(key, owner, KeyBudgetAlert::SoftBudget).await?;
    }
    host.key_model_budgets(key, owner).await
}

/// The key's team member budget: the member's spend in the key's team against the membership's
/// budget. A key without a member spend, user or team has none.
pub(crate) async fn team_member_budget<H: AuthHost>(
    host: &H,
    record: &KeyRecord,
    key: &KeyRow<H>,
    owner: Option<&UserRow<H>>,
    settings: &KeySettings,
) -> Result<(), Error> {
    let (Some(recorded), Some(user_id), Some(team_id)) = (
        record.team_member_spend,
        record.user_id.as_deref(),
        record.team_id.as_deref(),
    ) else {
        return Ok(());
    };
    if !settings.has_db || host.key_skips_budget_checks(key).await? {
        return Ok(());
    }
    let Some(max_budget) = host
        .team_member_max_budget(user_id, team_id)
        .await?
        .filter(|budget| *budget > 0.0)
    else {
        return Ok(());
    };
    let counter = SpendCounter::TeamMember {
        user_id,
        team_id,
        recorded,
        max_budget,
    };
    let spend = host.spend(counter).await?;
    if spend < max_budget {
        return Ok(());
    }
    host.key_budget_alert(key, owner, KeyBudgetAlert::TeamMember { spend, max_budget })
        .await?;
    let entity_id = format!("{user_id}:{team_id}");
    Err(BudgetExceeded {
        current_cost: spend,
        max_budget,
        message: format!(
            "Budget has been exceeded! TeamMember={entity_id} Current cost: {spend}, Max budget: {max_budget}"
        ),
        entity_type: Some("team_member".to_owned()),
        entity_id: Some(entity_id),
    }
    .into())
}

/// The proxy budget alert, sent with the proxy's global spend whenever the proxy has a budget and
/// a database to read the spend from.
pub(crate) async fn proxy_budget_alert<H: AuthHost>(
    host: &H,
    key: &KeyRow<H>,
    settings: &KeySettings,
) -> Result<(), Error> {
    if !(settings.proxy_budget && settings.has_db) {
        return Ok(());
    }
    match host.global_proxy_spend().await? {
        Some(spend) => host.key_budget_alert(key, None, KeyBudgetAlert::Proxy { spend }).await,
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;

    use super::*;
    use crate::testing::ScriptedHost;

    fn record() -> KeyRecord {
        KeyRecord {
            team_member_spend: Some(1.0),
            user_id: Some("u".to_owned()),
            team_id: Some("t".to_owned()),
            ..KeyRecord::default()
        }
    }

    fn settings() -> KeySettings {
        KeySettings {
            has_db: true,
            proxy_budget: true,
            ..KeySettings::default()
        }
    }

    #[rstest]
    #[case::under(3.0, Some(5.0), false, &["team_member_max_budget u t", "spend team_member u t"])]
    #[case::at_the_budget(5.0, Some(5.0), true, &["team_member_max_budget u t", "spend team_member u t", "key_budget_alert key team_member"])]
    #[case::no_member_budget(9.0, None, false, &["team_member_max_budget u t"])]
    #[case::zero_budget(9.0, Some(0.0), false, &["team_member_max_budget u t"])]
    #[tokio::test]
    async fn a_member_at_their_budget_is_alerted_then_denied(
        #[case] spend: f64,
        #[case] max_budget: Option<f64>,
        #[case] denied: bool,
        #[case] calls: &[&str],
    ) {
        let host = ScriptedHost {
            spends: HashMap::from([("team_member u t", spend)]),
            team_member_max_budget: max_budget,
            ..ScriptedHost::default()
        };
        let outcome = team_member_budget(&host, &record(), &"key".to_owned(), None, &settings()).await;
        assert_eq!(outcome.is_err(), denied);
        let mut expected = vec!["key_skips_budget_checks key"];
        expected.extend_from_slice(calls);
        assert_eq!(host.calls(), expected);
    }

    #[rstest]
    #[case::no_member_spend(KeyRecord { team_member_spend: None, ..record() })]
    #[case::no_team(KeyRecord { team_id: None, ..record() })]
    #[tokio::test]
    async fn a_key_without_a_team_member_reads_nothing(#[case] record: KeyRecord) {
        let host = ScriptedHost::default();
        assert_eq!(
            team_member_budget(&host, &record, &"key".to_owned(), None, &settings()).await,
            Ok(())
        );
        assert!(host.calls().is_empty());
    }

    #[tokio::test]
    async fn a_free_request_skips_the_member_budget() {
        let host = ScriptedHost {
            key_skips_budget_checks: true,
            ..ScriptedHost::default()
        };
        assert_eq!(
            team_member_budget(&host, &record(), &"key".to_owned(), None, &settings()).await,
            Ok(())
        );
        assert_eq!(host.calls(), ["key_skips_budget_checks key"]);
    }

    fn budgeted(budget: KeyBudgetRecord) -> KeyRecord {
        KeyRecord {
            token: Some("hashed".to_owned()),
            budget,
            ..KeyRecord::default()
        }
    }

    fn alerts(host: &ScriptedHost) -> Vec<String> {
        host.calls()
            .into_iter()
            .filter(|call| call.starts_with("key_budget_alert") || call.starts_with("throttle_key"))
            .collect()
    }

    #[rstest]
    #[case::default_share_reached(8.0, MaxBudgetWarning::Default { share: 0.8 }, true)]
    #[case::default_share_not_reached(7.0, MaxBudgetWarning::Default { share: 0.8 }, false)]
    #[case::default_over_budget_warns_no_more(10.0, MaxBudgetWarning::Default { share: 0.8 }, false)]
    #[case::emails_lowest_reached(5.0, MaxBudgetWarning::Emails { lowest_percent: Some(50.0) }, true)]
    #[case::emails_without_a_percent(9.0, MaxBudgetWarning::Emails { lowest_percent: None }, false)]
    fn the_max_budget_warning_follows_the_configured_threshold(
        #[case] spend: f64,
        #[case] warning: MaxBudgetWarning,
        #[case] due: bool,
    ) {
        let budget = KeyBudgetRecord {
            max_budget: Some(10.0),
            spend: Some(spend),
            warning,
            ..KeyBudgetRecord::default()
        };
        assert_eq!(max_budget_warning_due(&budget), due);
    }

    #[rstest]
    #[case::plain_name(Some("sk-...abcd"), "Key=ci (sk-...abcd)")]
    #[case::unmasked_name(Some("sk-live-secret"), "Key=ci ")]
    #[tokio::test]
    async fn an_over_budget_key_is_denied_naming_only_a_masked_key_name(
        #[case] key_name: Option<&str>,
        #[case] named: &str,
    ) {
        let host = ScriptedHost {
            spends: HashMap::from([("key hashed", 12.0)]),
            ..ScriptedHost::default()
        };
        let record = budgeted(KeyBudgetRecord {
            max_budget: Some(10.0),
            key_alias: Some("ci".to_owned()),
            key_name: key_name.map(str::to_owned),
            ..KeyBudgetRecord::default()
        });
        let Err(Error::BudgetExceeded(exceeded)) = key_max_budget(&host, &record, &"key".to_owned(), None).await else {
            panic!("expected the budget to deny");
        };
        assert!(exceeded.message.contains(named), "{}", exceeded.message);
        assert_eq!(alerts(&host), ["key_budget_alert key token_budget"]);
    }

    #[tokio::test]
    async fn a_key_that_opted_in_is_throttled_instead_of_refused() {
        let host = ScriptedHost {
            spends: HashMap::from([("key hashed", 12.0)]),
            ..ScriptedHost::default()
        };
        let record = budgeted(KeyBudgetRecord {
            max_budget: Some(10.0),
            throttle_share: Some(0.5),
            ..KeyBudgetRecord::default()
        });
        assert_eq!(key_max_budget(&host, &record, &"key".to_owned(), None).await, Ok(()));
        assert_eq!(alerts(&host), ["key_budget_alert key token_budget", "throttle_key key 0.5"]);
    }

    #[rstest]
    #[case::llm_route(true, &["key_budget_alert key token_budget", "key_budget_alert key soft_budget"])]
    #[case::management_route(false, &["key_budget_alert key soft_budget"])]
    #[tokio::test]
    async fn the_max_budget_applies_on_llm_routes_and_the_soft_budget_everywhere(
        #[case] is_llm_api_route: bool,
        #[case] expected: &[&str],
    ) {
        let host = ScriptedHost::default();
        let record = budgeted(KeyBudgetRecord {
            max_budget: Some(10.0),
            spend: Some(3.0),
            soft_budget: Some(2.0),
            ..KeyBudgetRecord::default()
        });
        assert_eq!(
            key_budgets(&host, &record, &"key".to_owned(), None, is_llm_api_route).await,
            Ok(())
        );
        assert_eq!(alerts(&host), expected);
        assert_eq!(host.calls().last().map(String::as_str), Some("key_model_budgets key -"));
    }

    #[tokio::test]
    async fn a_free_request_skips_every_key_budget() {
        let host = ScriptedHost {
            key_skips_budget_checks: true,
            ..ScriptedHost::default()
        };
        let record = budgeted(KeyBudgetRecord {
            max_budget: Some(1.0),
            ..KeyBudgetRecord::default()
        });
        assert_eq!(key_budgets(&host, &record, &"key".to_owned(), None, true).await, Ok(()));
        assert_eq!(host.calls(), ["key_skips_budget_checks key"]);
    }

    #[rstest]
    #[case::spend_read(Some(4.0), &["global_proxy_spend", "key_budget_alert key proxy"])]
    #[case::no_spend(None, &["global_proxy_spend"])]
    #[tokio::test]
    async fn the_proxy_alert_goes_out_with_the_global_spend(#[case] spend: Option<f64>, #[case] calls: &[&str]) {
        let host = ScriptedHost {
            global_proxy_spend: spend,
            ..ScriptedHost::default()
        };
        assert_eq!(proxy_budget_alert(&host, &"key".to_owned(), &settings()).await, Ok(()));
        assert_eq!(host.calls(), calls);
    }
}
