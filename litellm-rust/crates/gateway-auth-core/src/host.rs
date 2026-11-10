use std::{error::Error, future::Future};

use crate::{
    answers::{
        FallbackTarget, Introspection, KeyFacts, ModelBudgetScope, OwnerFacts, PresentedKey,
        SigningKey, TeamMember, TeamMemberBudget, TeamMemberModelBudgetScope,
    },
    identity::{CustomAuthResult, JwtIdentity},
    inputs::RequestFacts,
    outcome::ModelBudgetExceeded,
};

pub trait ModelBudgetHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn read_model_budget_scope(
        &self,
    ) -> impl Future<Output = Result<ModelBudgetScope, Self::Error>> + Send;
    fn check_key_model_budget(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Option<ModelBudgetExceeded>, Self::Error>> + Send;
    fn authorize_budget_fallback(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;
    fn rewrite_request_model(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn check_user_model_budget(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Option<ModelBudgetExceeded>, Self::Error>> + Send;
}

pub trait TeamMemberModelBudgetHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn read_team_member_model_budget_scope(
        &self,
    ) -> impl Future<Output = Result<TeamMemberModelBudgetScope, Self::Error>> + Send;
    fn check_team_member_model_budget(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Option<ModelBudgetExceeded>, Self::Error>> + Send;
    fn read_team_member_fallbacks(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Vec<FallbackTarget>, Self::Error>> + Send;
    fn vet_team_member_fallback(
        &self,
        target: &FallbackTarget,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;
    fn reroute_team_member_request(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

pub trait KeyHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn resolve_end_user(
        &self,
        prefetch_key: bool,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn resolve_presented_key(
        &self,
        hashed_token: &str,
    ) -> impl Future<Output = Result<Option<PresentedKey>, Self::Error>> + Send;
    fn evict_presented_key(
        &self,
        hashed_token: &str,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn load_key(&self, hashed_token: &str) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn prepare_key(
        &self,
        loaded_from_db: bool,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn inspect_key(&self) -> impl Future<Output = Result<KeyFacts, Self::Error>> + Send;
    fn load_key_owner(
        &self,
        user_id: &str,
    ) -> impl Future<Output = Result<Option<OwnerFacts>, Self::Error>> + Send;
}

pub trait AuthorizeHost: Sync {
    type Error: Error + Send + Sync + 'static;
    type ModelBudgets: ModelBudgetHost;

    fn model_budgets(&self) -> Self::ModelBudgets;

    fn enforce_model_access(&self) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn check_budget_exemption(&self) -> impl Future<Output = Result<bool, Self::Error>> + Send;
    fn read_team_member_budget(
        &self,
        member: &TeamMember,
    ) -> impl Future<Output = Result<Option<TeamMemberBudget>, Self::Error>> + Send;
    fn alert_team_member_budget(
        &self,
        member: &TeamMember,
        budget: TeamMemberBudget,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn send_key_budget_alerts(&self) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn read_key_spend(
        &self,
        max_budget: f64,
    ) -> impl Future<Output = Result<f64, Self::Error>> + Send;
    fn alert_key_budget(&self, spend: f64) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn alert_key_soft_budget(&self) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn load_key_context(&self) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn read_proxy_spend(&self) -> impl Future<Output = Result<Option<f64>, Self::Error>> + Send;
    fn alert_proxy_budget(
        &self,
        spend: f64,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
}

pub trait JwtHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn signing_keys(
        &self,
        issuer: &str,
    ) -> impl Future<Output = Result<Vec<SigningKey>, Self::Error>> + Send;
    fn mapped_key_hash(
        &self,
        identity: &JwtIdentity,
    ) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send;
}

pub trait OAuth2IntrospectionHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn introspect(
        &self,
        token: &str,
    ) -> impl Future<Output = Result<Introspection, Self::Error>> + Send;
}

pub trait OAuth2ProxyHost {
    fn client_ip(&self) -> Option<&str>;
    fn header(&self, name: &str) -> Option<&str>;
}

pub trait CustomAuthHost: Sync {
    type Error: Error + Send + Sync + 'static;

    fn custom_auth(
        &self,
        facts: &RequestFacts,
    ) -> impl Future<Output = Result<CustomAuthResult, Self::Error>> + Send;
}
