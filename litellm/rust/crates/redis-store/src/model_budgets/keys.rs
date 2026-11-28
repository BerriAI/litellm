//! Scope-specific wire keys for model budgets.

use strum::EnumProperty;

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    strum::IntoStaticStr,
    strum::EnumProperty,
    strum::VariantArray,
)]
pub enum Scope {
    #[strum(
        serialize = "virtual_key_spend",
        props(start = "virtual_key_budget_start_time")
    )]
    Key,
    #[strum(
        serialize = "user_model_spend",
        props(start = "user_model_budget_start_time")
    )]
    User,
    #[strum(
        serialize = "end_user_model_spend",
        props(start = "end_user_budget_start_time")
    )]
    EndUser,
    #[strum(
        serialize = "team_model_spend",
        props(start = "team_model_budget_start_time")
    )]
    Team,
    #[strum(
        serialize = "team_member_model_spend",
        props(start = "team_member_model_budget_start_time")
    )]
    TeamMember,
}

/// The identity of a model budget in the shared proxy cache.
pub struct Counter<'a> {
    pub scope: Scope,
    pub id: &'a str,
    pub model: &'a str,
    pub duration: &'a str,
}

impl Counter<'_> {
    pub fn spend(&self) -> String {
        self.key(self.scope.into())
    }

    pub(super) fn start(&self) -> String {
        self.key(
            self.scope
                .get_str("start")
                .expect("each scope has a start prefix"),
        )
    }

    fn key(&self, prefix: &str) -> String {
        format!("{prefix}:{}:{}:{}", self.id, self.model, self.duration)
    }
}
