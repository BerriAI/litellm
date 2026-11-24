use litellm_python_compat::{number::PyNumber, unicode::isspace};
use serde::{Deserialize, Serialize};

fn is_masked_key_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("sk-...") else {
        return false;
    };
    rest.is_empty()
        || (rest.chars().count() == 4
            && rest
                .chars()
                .all(|c| !isspace(c) && !matches!(c, '\u{0}'..='\u{1f}' | '\u{7f}'..='\u{9f}')))
}

fn py_str(value: Option<&str>) -> &str {
    value.unwrap_or("None")
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "check", rename_all = "snake_case", deny_unknown_fields)]
pub enum Check {
    KeyMax {
        spend: PyNumber,
        max_budget: PyNumber,
        token: Option<String>,
        key_alias: Option<String>,
        key_name: Option<String>,
        throttle_pct: Option<PyNumber>,
    },
    KeyWindow {
        spend: PyNumber,
        max_budget: PyNumber,
        budget_duration: String,
        token: Option<String>,
    },
    KeySoft {
        spend: PyNumber,
        soft_budget: Option<PyNumber>,
    },
    TeamMax {
        spend: PyNumber,
        max_budget: PyNumber,
        team_id: Option<String>,
    },
    TeamWindow {
        spend: PyNumber,
        max_budget: PyNumber,
        budget_duration: String,
        team_id: Option<String>,
    },
    UserWindow {
        spend: PyNumber,
        max_budget: PyNumber,
        budget_duration: String,
        user_id: String,
    },
    TeamSoft {
        spend: Option<PyNumber>,
        soft_budget: Option<PyNumber>,
    },
    ProjectMax {
        spend: PyNumber,
        max_budget: PyNumber,
        project_id: Option<String>,
    },
    ProjectSoft {
        spend: Option<PyNumber>,
        soft_budget: Option<PyNumber>,
    },
    Organization {
        spend: PyNumber,
        max_budget: PyNumber,
        org_id: String,
    },
    Tag {
        spend: PyNumber,
        max_budget: PyNumber,
        tag: String,
    },
    ModelAccessGroup {
        spend: PyNumber,
        max_budget: PyNumber,
        group: String,
    },
    User {
        spend: PyNumber,
        max_budget: PyNumber,
        user_id: Option<String>,
    },
    TeamMember {
        spend: PyNumber,
        max_budget: PyNumber,
        user_id: Option<String>,
        team_id: Option<String>,
    },
    TeamMemberPositive {
        spend: PyNumber,
        max_budget: PyNumber,
        user_id: Option<String>,
        team_id: Option<String>,
    },
    EndUser {
        spend: PyNumber,
        max_budget: PyNumber,
        end_user_id: Option<String>,
    },
    Proxy {
        spend: PyNumber,
        max_budget: PyNumber,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Refusal {
    pub current_cost: PyNumber,
    pub max_budget: PyNumber,
    pub message: String,
    pub entity_type: &'static str,
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    Skip,
    Pass,
    Alert,
    Throttle { pct: PyNumber },
    Refuse(Refusal),
}

fn refuse(
    spend: PyNumber,
    max_budget: PyNumber,
    message: String,
    entity_type: &'static str,
    entity_id: Option<String>,
) -> Decision {
    Decision::Refuse(Refusal {
        current_cost: spend,
        max_budget,
        message,
        entity_type,
        entity_id,
    })
}

fn current(spend: PyNumber, max_budget: PyNumber, label: &str) -> String {
    format!(
        "Budget has been exceeded! {label} Current cost: {}, Max budget: {}",
        spend.text(),
        max_budget.text()
    )
}

fn window(spend: PyNumber, max_budget: PyNumber) -> String {
    format!("Spend=${}, Limit=${}", spend.fixed(4), max_budget.fixed(2))
}

fn at_or_over(spend: PyNumber, max_budget: PyNumber) -> bool {
    max_budget.value().is_finite() && spend.value() >= max_budget.value()
}

fn over(spend: PyNumber, max_budget: PyNumber) -> bool {
    max_budget.value().is_finite() && spend.value() > max_budget.value()
}

fn crossed(spend: Option<PyNumber>, soft_budget: Option<PyNumber>) -> Decision {
    match (spend, soft_budget) {
        (Some(spend), Some(soft)) if spend.value() >= soft.value() => Decision::Alert,
        _ => Decision::Pass,
    }
}

pub fn decide(check: &Check) -> Decision {
    match check.clone() {
        Check::KeyMax {
            spend,
            max_budget,
            token,
            key_alias,
            key_name,
            throttle_pct,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            if let Some(pct) = throttle_pct {
                return Decision::Throttle { pct };
            }
            let label = key_alias.filter(|alias| !alias.is_empty());
            let label = label.as_deref().unwrap_or("key");
            let descriptor = match key_name.as_deref() {
                Some(name) if !name.is_empty() && is_masked_key_name(name) => {
                    format!("{label} ({name})")
                }
                _ => label.to_owned(),
            };
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("Key={descriptor}")),
                "key",
                token,
            )
        }
        Check::KeyWindow {
            spend,
            max_budget,
            budget_duration,
            token,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "ExceededBudget: Key over {budget_duration} budget. {}",
                    window(spend, max_budget)
                ),
                "key",
                token,
            )
        }
        Check::KeySoft { spend, soft_budget } => match soft_budget {
            Some(soft) if soft.value() != 0.0 && spend.value() >= soft.value() => Decision::Alert,
            _ => Decision::Pass,
        },
        Check::TeamMax {
            spend,
            max_budget,
            team_id,
        } => {
            if !over(spend, max_budget) {
                return Decision::Pass;
            }
            let label = format!("Team={}", py_str(team_id.as_deref()));
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &label),
                "team",
                team_id,
            )
        }
        Check::TeamWindow {
            spend,
            max_budget,
            budget_duration,
            team_id,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "ExceededBudget: Team={} over {budget_duration} budget. {}",
                    py_str(team_id.as_deref()),
                    window(spend, max_budget)
                ),
                "team",
                team_id,
            )
        }
        Check::UserWindow {
            spend,
            max_budget,
            budget_duration,
            user_id,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "ExceededBudget: User={user_id} over {budget_duration} budget. {}",
                    window(spend, max_budget)
                ),
                "user",
                Some(user_id),
            )
        }
        Check::TeamSoft { spend, soft_budget } | Check::ProjectSoft { spend, soft_budget } => {
            crossed(spend, soft_budget)
        }
        Check::ProjectMax {
            spend,
            max_budget,
            project_id,
        } => {
            if !max_budget.value().is_finite() {
                return Decision::Skip;
            }
            if spend.value() < max_budget.value() || spend.value().is_nan() {
                return Decision::Pass;
            }
            let label = format!("Project={}", py_str(project_id.as_deref()));
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &label),
                "project",
                project_id,
            )
        }
        Check::Organization {
            spend,
            max_budget,
            org_id,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("Organization={org_id}")),
                "organization",
                Some(org_id),
            )
        }
        Check::Tag {
            spend,
            max_budget,
            tag,
        } => {
            if spend.value() <= max_budget.value() {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("Tag={tag}")),
                "tag",
                Some(tag),
            )
        }
        Check::ModelAccessGroup {
            spend,
            max_budget,
            group,
        } => {
            if max_budget.value() <= 0.0 {
                return Decision::Skip;
            }
            if spend.value() < max_budget.value() {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("Model access group={group}")),
                "model_access_group",
                Some(group),
            )
        }
        Check::User {
            spend,
            max_budget,
            user_id,
        } => {
            if !at_or_over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "ExceededBudget: User={} over budget. Spend={}, Budget={}",
                    py_str(user_id.as_deref()),
                    spend.text(),
                    max_budget.text()
                ),
                "user",
                user_id,
            )
        }
        Check::TeamMember {
            spend,
            max_budget,
            user_id,
            team_id,
        } => {
            if !max_budget.value().is_finite() {
                return Decision::Skip;
            }
            if spend.value() < max_budget.value() || spend.value().is_nan() {
                return Decision::Pass;
            }
            let (user, team) = (py_str(user_id.as_deref()), py_str(team_id.as_deref()));
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("User={user} in Team={team}")),
                "team_member",
                Some(format!("{user}:{team}")),
            )
        }
        Check::TeamMemberPositive {
            spend,
            max_budget,
            user_id,
            team_id,
        } => {
            if spend.value() < max_budget.value()
                || spend.value().is_nan()
                || max_budget.value().is_nan()
            {
                return Decision::Pass;
            }
            let entity = format!(
                "{}:{}",
                py_str(user_id.as_deref()),
                py_str(team_id.as_deref())
            );
            refuse(
                spend,
                max_budget,
                current(spend, max_budget, &format!("TeamMember={entity}")),
                "team_member",
                Some(entity),
            )
        }
        Check::EndUser {
            spend,
            max_budget,
            end_user_id,
        } => {
            if spend.value() <= max_budget.value()
                || spend.value().is_nan()
                || max_budget.value().is_nan()
            {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "ExceededBudget: End User={} over budget. Spend={}, Budget={}",
                    py_str(end_user_id.as_deref()),
                    spend.text(),
                    max_budget.text()
                ),
                "end_user",
                end_user_id,
            )
        }
        Check::Proxy { spend, max_budget } => {
            if !over(spend, max_budget) {
                return Decision::Pass;
            }
            refuse(
                spend,
                max_budget,
                format!(
                    "Budget has been exceeded! Current cost: {}, Max budget: {}",
                    spend.text(),
                    max_budget.text()
                ),
                "proxy",
                None,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("sk-...", true)]
    #[case("sk-...abcd", true)]
    #[case("sk-...abc", false)]
    #[case("sk-...ab d", false)]
    #[case("sk-...abcde", false)]
    #[case("sk-...ab\u{85}d", false)]
    #[case("sk-...ab\u{3000}d", false)]
    #[case("sk-...ab\u{202f}d", false)]
    #[case("sk-...ab\u{2028}d", false)]
    #[case("sk-...ab\u{200b}d", true)]
    #[case("sk-...ab\u{180e}d", true)]
    #[case("sk-...ab\u{feff}d", true)]
    #[case("my-key", false)]
    fn masked_key_names_match_the_proxy_pattern(#[case] name: &str, #[case] masked: bool) {
        assert_eq!(is_masked_key_name(name), masked);
    }
}
