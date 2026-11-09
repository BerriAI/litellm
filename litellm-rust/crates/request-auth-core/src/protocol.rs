use std::collections::{BTreeMap, BTreeSet};

use litellm_secret_value::SecretValue;
use serde::{Deserialize, Serialize};

use crate::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DateTime {
    pub wall_microseconds: i64,
    pub offset_microseconds: Option<i64>,
    pub timezone_id: u64,
    pub reference: usize,
}

impl DateTime {
    pub fn before(&self, other: &Self) -> Result<bool, Error> {
        if self.timezone_id == other.timezone_id {
            return Ok(self.wall_microseconds < other.wall_microseconds);
        }
        match (self.offset_microseconds, other.offset_microseconds) {
            (Some(left), Some(right)) => {
                Ok(self.wall_microseconds - left < other.wall_microseconds - right)
            }
            (None, None) => Ok(self.wall_microseconds < other.wall_microseconds),
            _ => Err(Error::IncompatibleDateTimes),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RequestFacts {
    pub route: String,
    pub token: Option<SecretValue>,
    pub is_llm_api_route: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    pub master_key: Option<SecretValue>,
    pub allow_user_auth: bool,
    pub has_db: bool,
    pub master_key_only_routes: BTreeSet<String>,
    pub proxy_max_budget: f64,
}

#[derive(Clone, veil::Redact, Serialize)]
#[serde(tag = "kind")]
pub enum Grant {
    NoMasterKey {
        #[redact(with = "[REDACTED]")]
        token: Option<String>,
    },
    UserAuthRoute,
    MasterKey,
    CachedAdminKey,
    VirtualKey {
        budget_throttle_pct: Option<f64>,
    },
    WithinModelBudgets,
}

#[derive(Clone, veil::Redact, Serialize)]
#[serde(tag = "kind")]
pub enum Denial {
    ModelAccessDenied {
        decision: crate::model_access::Decision,
    },
    MissingKey,
    MalformedKey,
    UserAuthDisabled,
    ExpiredKey {
        expires: DateTime,
        now: DateTime,
        #[redact(with = "[REDACTED]")]
        key_label: String,
    },
    MasterKeyOnlyRoute,
    NoDatabase,
    NotAVirtualKey,
    KeyBlocked,
    OwnerDeactivated {
        user_id: String,
    },
    TeamMemberBudgetExceeded {
        user_id: String,
        team_id: String,
        spend: f64,
        max_budget: f64,
    },
    KeyBudgetExceeded {
        spend: f64,
        max_budget: f64,
    },
    KeyWithoutToken,
    ModelBudgetExceeded {
        message: String,
        current_cost: f64,
        max_budget: f64,
        entity_type: Option<String>,
        entity_id: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelBudgetExceeded {
    pub message: String,
    pub current_cost: f64,
    pub max_budget: f64,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelBudgetRead {
    pub snapshot: crate::model_budget_data::Snapshot,
    pub message: String,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
}

impl ModelBudgetRead {
    pub fn exceeded(self) -> Option<ModelBudgetExceeded> {
        let result = crate::model_budget_data::check(&self.snapshot);
        if !result.denied {
            return None;
        }
        Some(ModelBudgetExceeded {
            message: self.message,
            current_cost: result.current_spend,
            max_budget: self.snapshot.max_budget?,
            entity_type: self.entity_type,
            entity_id: self.entity_id,
        })
    }
}

impl From<ModelBudgetExceeded> for Denial {
    fn from(value: ModelBudgetExceeded) -> Self {
        Self::ModelBudgetExceeded {
            message: value.message,
            current_cost: value.current_cost,
            max_budget: value.max_budget,
            entity_type: value.entity_type,
            entity_id: value.entity_id,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum PresentedKey {
    AdminKey { expires: Option<DateTime> },
    NonAdminKey,
}

#[derive(Clone, Debug, Deserialize)]
pub struct KeyFacts {
    pub has_token: bool,
    pub blocked: bool,
    pub expires: Option<DateTime>,
    pub spend: f64,
    pub soft_budget: Option<f64>,
    pub max_budget: Option<f64>,
    pub team_member_spend: Option<f64>,
    pub user_id: Option<String>,
    pub team_id: Option<String>,
    pub budget_throttle_pct: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OwnerFacts {
    pub deactivated: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TeamMemberBudget {
    pub spend: f64,
    pub max_budget: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelBudgetScope {
    pub models: Vec<String>,
    pub rewritable_model: Option<String>,
    pub key_has_model_budgets: bool,
    pub key_budget_fallbacks: BTreeMap<String, Vec<String>>,
    pub user_has_model_budgets: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TeamMemberModelBudgetScope {
    pub models: Vec<String>,
    pub rewritable_model: Option<String>,
    pub has_model_budgets: bool,
    pub can_fall_back: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FallbackTarget {
    pub requested: String,
    pub model: String,
    pub served_model: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind")]
pub enum Effect {
    ResolveEndUser {
        prefetch_key: bool,
    },
    ResolvePresentedKey {
        hashed_token: String,
    },
    EvictPresentedKey {
        hashed_token: String,
    },
    LoadKey {
        hashed_token: String,
    },
    PrepareKey {
        loaded_from_db: bool,
    },
    InspectKey,
    EnforceModelAccess,
    PrefetchAuthObjects,
    LoadKeyOwner {
        user_id: String,
    },
    CheckBudgetExemption,
    ReadTeamMemberBudget {
        user_id: String,
        team_id: String,
    },
    AlertTeamMemberBudget {
        user_id: String,
        team_id: String,
        budget: TeamMemberBudget,
    },
    SendKeyBudgetAlerts,
    ReadKeySpend {
        max_budget: f64,
    },
    AlertKeyBudget {
        spend: f64,
    },
    AlertKeySoftBudget,
    LoadKeyContext,
    ReadProxySpend,
    AlertProxyBudget {
        spend: f64,
    },
    ReadModelBudgetScope,
    CheckKeyModelBudget {
        model: String,
    },
    AuthorizeBudgetFallback {
        model: String,
    },
    RewriteRequestModel {
        model: String,
    },
    CheckUserModelBudget {
        model: String,
    },
    ReadTeamMemberModelBudgetScope,
    CheckTeamMemberModelBudget {
        model: String,
    },
    ReadTeamMemberFallbacks {
        model: String,
    },
    VetTeamMemberFallback {
        target: FallbackTarget,
    },
    RerouteTeamMemberRequest {
        model: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum Outcome {
    Allow { grant: Grant },
    Deny { denial: Denial },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Node {
    Outcome(Outcome),
    Effect(Effect),
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "value")]
pub enum Answer {
    ModelAccess(Box<crate::key_model_access::Facts>),
    FallbackAccess(Box<crate::fallback_access::Facts>),
    Candidate(Box<crate::fallback_candidate::Facts>),
    Unit,
    Bool(bool),
    Exemption(crate::budget_pricing::ExemptionFacts),
    Presented(Option<PresentedKey>),
    Key(KeyFacts),
    Owner(Option<OwnerFacts>),
    TeamBudget(Option<TeamMemberBudget>),
    Spend(f64),
    ProxySpend(Option<f64>),
    ModelScope(ModelBudgetScope),
    Exceeded(Option<ModelBudgetExceeded>),
    ModelBudgetData(ModelBudgetRead),
    TeamScope(TeamMemberModelBudgetScope),
    Targets(Vec<FallbackTarget>),
}

pub trait FromAnswer: Sized {
    fn from_answer(answer: Answer) -> Result<Self, Error>;
}

macro_rules! answer {
    ($type:ty, $variant:ident) => {
        impl FromAnswer for $type {
            fn from_answer(answer: Answer) -> Result<Self, Error> {
                match answer {
                    Answer::$variant(value) => Ok(value),
                    _ => Err(Error::InvalidAnswer),
                }
            }
        }
    };
}
answer!(bool, Bool);
answer!(crate::budget_pricing::ExemptionFacts, Exemption);
answer!(Option<PresentedKey>, Presented);
answer!(KeyFacts, Key);
answer!(Option<OwnerFacts>, Owner);
answer!(Option<TeamMemberBudget>, TeamBudget);
answer!(f64, Spend);
answer!(Option<f64>, ProxySpend);
answer!(ModelBudgetScope, ModelScope);
answer!(Option<ModelBudgetExceeded>, Exceeded);
answer!(ModelBudgetRead, ModelBudgetData);
answer!(TeamMemberModelBudgetScope, TeamScope);
answer!(Vec<FallbackTarget>, Targets);

impl FromAnswer for crate::fallback_candidate::Facts {
    fn from_answer(answer: Answer) -> Result<Self, Error> {
        match answer {
            Answer::Candidate(value) => Ok(*value),
            _ => Err(Error::InvalidAnswer),
        }
    }
}

impl FromAnswer for crate::key_model_access::Facts {
    fn from_answer(answer: Answer) -> Result<Self, Error> {
        match answer {
            Answer::ModelAccess(value) => Ok(*value),
            _ => Err(Error::InvalidAnswer),
        }
    }
}

impl FromAnswer for crate::fallback_access::Facts {
    fn from_answer(answer: Answer) -> Result<Self, Error> {
        match answer {
            Answer::FallbackAccess(value) => Ok(*value),
            _ => Err(Error::InvalidAnswer),
        }
    }
}

impl FromAnswer for () {
    fn from_answer(answer: Answer) -> Result<Self, Error> {
        match answer {
            Answer::Unit => Ok(()),
            _ => Err(Error::InvalidAnswer),
        }
    }
}

pub type Continuation = Box<dyn FnOnce(Answer) -> Result<Step, Error> + Send + Sync>;
pub type Then = Box<dyn FnOnce() -> Result<Step, Error> + Send + Sync>;

pub struct Step {
    pub node: Node,
    pub resume: Option<Continuation>,
}

impl Step {
    pub fn allow(grant: Grant) -> Result<Self, Error> {
        Ok(Self {
            node: Node::Outcome(Outcome::Allow { grant }),
            resume: None,
        })
    }
    pub fn deny(denial: impl Into<Denial>) -> Result<Self, Error> {
        Ok(Self {
            node: Node::Outcome(Outcome::Deny {
                denial: denial.into(),
            }),
            resume: None,
        })
    }
    pub fn effect<T: FromAnswer>(
        effect: Effect,
        resume: impl FnOnce(T) -> Result<Self, Error> + Send + Sync + 'static,
    ) -> Result<Self, Error> {
        Ok(Self {
            node: Node::Effect(effect),
            resume: Some(Box::new(move |answer| resume(T::from_answer(answer)?))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::same_zone_ignores_offset(10, Some(100), 1, 11, Some(200), 1, Ok(true))]
    #[case::different_zones_use_offset(10, Some(100), 1, 11, Some(200), 2, Ok(false))]
    #[case::equal_is_not_expired(10, None, 0, 10, None, 0, Ok(false))]
    #[case::mixed_naive_aware(10, None, 0, 10, Some(0), 1, Err(()))]
    #[allow(clippy::too_many_arguments)]
    fn expiry_comparison(
        #[case] wall: i64,
        #[case] offset: Option<i64>,
        #[case] zone: u64,
        #[case] other_wall: i64,
        #[case] other_offset: Option<i64>,
        #[case] other_zone: u64,
        #[case] expected: Result<bool, ()>,
    ) {
        let expires = DateTime {
            wall_microseconds: wall,
            offset_microseconds: offset,
            timezone_id: zone,
            reference: 0,
        };
        let now = DateTime {
            wall_microseconds: other_wall,
            offset_microseconds: other_offset,
            timezone_id: other_zone,
            reference: 1,
        };
        assert_eq!(expires.before(&now).map_err(|_| ()), expected);
    }

    #[rstest]
    fn wrong_answer_never_advances_the_decision() {
        let step = Step::effect(Effect::CheckBudgetExemption, |_: bool| {
            Step::allow(Grant::MasterKey)
        })
        .unwrap();
        assert!(matches!(
            (step.resume.unwrap())(Answer::Unit),
            Err(Error::InvalidAnswer)
        ));
    }
}
