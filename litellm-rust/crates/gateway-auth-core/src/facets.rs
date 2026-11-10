use std::{future::Future, time::SystemTime};

use crate::{
    host::{
        AuthorizeHost, CustomAuthHost, JwtHost, KeyHost, ModelBudgetHost, OAuth2IntrospectionHost,
        OAuth2ProxyHost, RequestCheckHost, TeamMemberModelBudgetHost,
    },
    identity::{
        Anonymous, AnonymousReason, AuthDenial, CustomAuthResult, Identity, KeyAuthentication,
        KeyDenial, KeyIdentity, OAuth2Denial, OAuth2Identity, OAuth2ProxyDenial,
        OAuth2ProxyIdentity, VirtualKey,
    },
    inputs::{
        AuthorizeSettings, CheckFacts, CheckSettings, JwtSettings, KeySettings,
        OAuth2IntrospectionSettings, OAuth2ProxySettings, PlanFacts, PlanSettings, RequestFacts,
    },
    outcome::{ModelBudgetResult, Outcome, RequestCheck},
    validators::{CredentialPlan, KeyPlan, Plan},
};

pub trait KeyCore: Sync {
    fn authenticate<H: KeyHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &KeySettings,
        now: SystemTime,
    ) -> impl Future<Output = Result<KeyAuthentication, H::Error>> + Send;
    fn verify<H: KeyHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        now: SystemTime,
    ) -> impl Future<Output = Result<Result<VirtualKey, KeyDenial>, H::Error>> + Send;
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
    fn plan(&self, facts: &PlanFacts, settings: &PlanSettings) -> Plan;
    fn after_custom_auth(
        &self,
        facts: &PlanFacts,
        settings: &PlanSettings,
        issued_key: bool,
    ) -> CredentialPlan;
    fn after_jwt(&self, facts: &PlanFacts, settings: &PlanSettings, issued_key: bool) -> KeyPlan;
    fn oauth2_introspection_applies(&self, facts: &PlanFacts, settings: &PlanSettings) -> bool;
}

pub trait AnonymousCore: Sync {
    fn authenticate(&self, reason: AnonymousReason) -> Anonymous;
}

pub trait JwtCore: Sync {
    fn authenticate<H: JwtHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
        settings: &JwtSettings,
        now: SystemTime,
    ) -> impl Future<Output = Result<Result<Identity, AuthDenial>, H::Error>> + Send;
}

pub trait OAuth2Core: Sync {
    fn introspect<H: OAuth2IntrospectionHost>(
        &self,
        host: &H,
        token: &str,
        settings: &OAuth2IntrospectionSettings,
    ) -> impl Future<Output = Result<Result<OAuth2Identity, OAuth2Denial>, H::Error>> + Send;
    fn proxy<H: OAuth2ProxyHost>(
        &self,
        host: &H,
        settings: &OAuth2ProxySettings,
    ) -> Result<OAuth2ProxyIdentity, OAuth2ProxyDenial>;
}

pub trait CustomAuthCore: Sync {
    fn authenticate<H: CustomAuthHost>(
        &self,
        host: &H,
        facts: &RequestFacts,
    ) -> impl Future<Output = Result<CustomAuthResult, H::Error>> + Send;
}

pub trait AuthorizeCore: Sync {
    fn authorize<H: AuthorizeHost>(
        &self,
        host: &H,
        identity: &KeyIdentity,
        facts: &RequestFacts,
        settings: &AuthorizeSettings,
    ) -> impl Future<Output = Result<Outcome, H::Error>> + Send;
    fn check_request<H: RequestCheckHost>(
        &self,
        host: &H,
        facts: &CheckFacts,
        settings: &CheckSettings,
    ) -> impl Future<Output = Result<RequestCheck, H::Error>> + Send;
}

pub trait AuthCore {
    type Keys: KeyCore;
    type ModelBudgets: ModelBudgetCore;
    type Planner: AuthPlanner;
    type Anonymous: AnonymousCore;
    type Jwt: JwtCore;
    type OAuth2: OAuth2Core;
    type Custom: CustomAuthCore;
    type Authorize: AuthorizeCore;

    fn keys(&self) -> &Self::Keys;
    fn model_budgets(&self) -> &Self::ModelBudgets;
    fn plan(&self) -> &Self::Planner;
    fn anonymous(&self) -> &Self::Anonymous;
    fn jwt(&self) -> &Self::Jwt;
    fn oauth2(&self) -> &Self::OAuth2;
    fn custom(&self) -> &Self::Custom;
    fn authorize(&self) -> &Self::Authorize;
}
