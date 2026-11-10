use std::{future::Future, time::SystemTime};

use crate::{
    host::{
        AuthorizeHost, CustomAuthHost, JwtHost, KeyHost, ModelBudgetHost, OAuth2Host,
        TeamMemberModelBudgetHost,
    },
    inputs::{
        AuthorizeSettings, JwtSettings, KeySettings, OAuth2Settings, PlanSettings, RequestFacts,
    },
    outcome::{
        AnonymousReason, CustomIdentity, Denial, Identity, ModelBudgetResult, OAuth2Denial,
        OAuth2Identity, Outcome, Validator,
    },
};

pub trait KeyCore: Sync {
    fn authenticate<H: KeyHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &KeySettings,
        now: SystemTime,
    ) -> impl Future<Output = Result<Outcome, H::Error>> + Send;
    fn check<H: KeyHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &KeySettings,
        now: SystemTime,
    ) -> impl Future<Output = Result<Outcome, H::Error>> + Send;
}

pub trait ModelBudgetCore: Sync {
    fn check_key_and_user<H: ModelBudgetHost>(
        &self,
        host: &H,
    ) -> impl Future<Output = Result<ModelBudgetResult, H::Error>> + Send;
    fn check_user<H: ModelBudgetHost>(
        &self,
        host: &H,
    ) -> impl Future<Output = Result<ModelBudgetResult, H::Error>> + Send;
    fn check_team_member<H: TeamMemberModelBudgetHost>(
        &self,
        host: &H,
    ) -> impl Future<Output = Result<ModelBudgetResult, H::Error>> + Send;
}

pub trait AuthPlanner: Sync {
    fn plan(&self, facts: &RequestFacts, settings: &PlanSettings) -> Validator;
}

pub trait AnonymousCore: Sync {
    fn authenticate(&self, facts: &RequestFacts, settings: &PlanSettings) -> AnonymousReason;
}

pub trait JwtCore: Sync {
    fn authenticate<H: JwtHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &JwtSettings,
        now: SystemTime,
    ) -> impl Future<Output = Result<Result<Identity, Denial>, H::Error>> + Send;
}

pub trait OAuth2Core: Sync {
    fn authenticate<H: OAuth2Host>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &OAuth2Settings,
    ) -> impl Future<Output = Result<Result<OAuth2Identity, OAuth2Denial>, H::Error>> + Send;
}

pub trait CustomAuthCore: Sync {
    fn authenticate<H: CustomAuthHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
    ) -> impl Future<Output = Result<CustomIdentity, H::Error>> + Send;
}

pub trait AuthorizeCore: Sync {
    fn authorize<H: AuthorizeHost>(
        &self,
        host: &H,
        identity: &Identity,
        facts: &RequestFacts,
        settings: &AuthorizeSettings,
    ) -> impl Future<Output = Result<Outcome, H::Error>> + Send;
}

pub trait AuthCore {
    type Keys: KeyCore;
    type ModelBudgets: ModelBudgetCore;

    fn keys(&self) -> &Self::Keys;
    fn model_budgets(&self) -> &Self::ModelBudgets;
}

pub trait TargetAuthCore: AuthCore {
    type Planner: AuthPlanner;
    type Anonymous: AnonymousCore;
    type Jwt: JwtCore;
    type OAuth2: OAuth2Core;
    type Custom: CustomAuthCore;
    type Authorize: AuthorizeCore;

    fn plan(&self) -> &Self::Planner;
    fn anonymous(&self) -> &Self::Anonymous;
    fn jwt(&self) -> &Self::Jwt;
    fn oauth2(&self) -> &Self::OAuth2;
    fn custom(&self) -> &Self::Custom;
    fn authorize(&self) -> &Self::Authorize;
}
