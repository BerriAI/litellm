use std::time::SystemTime;

use crate::{
    facets::{
        AnonymousCore, AuthCore, AuthPlanner, AuthorizeCore, CustomAuthCore, JwtCore, KeyCore,
        ModelBudgetCore, OAuth2Core,
    },
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

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCore;

impl AuthCore for NativeCore {
    type Keys = Unported;
    type ModelBudgets = Unported;
    type Planner = Unported;
    type Anonymous = Unported;
    type Jwt = Unported;
    type OAuth2 = Unported;
    type Custom = Unported;
    type Authorize = Unported;

    fn keys(&self) -> &Unported {
        &Unported
    }

    fn model_budgets(&self) -> &Unported {
        &Unported
    }

    fn plan(&self) -> &Unported {
        &Unported
    }

    fn anonymous(&self) -> &Unported {
        &Unported
    }

    fn jwt(&self) -> &Unported {
        &Unported
    }

    fn oauth2(&self) -> &Unported {
        &Unported
    }

    fn custom(&self) -> &Unported {
        &Unported
    }

    fn authorize(&self) -> &Unported {
        &Unported
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Unported;

const RUNS_IN_PYTHON: &str = "every auth facet runs in PythonCore";

impl KeyCore for Unported {
    async fn authenticate<H: KeyHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _settings: &KeySettings,
        _now: SystemTime,
    ) -> Result<KeyAuthentication, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    async fn verify<H: KeyHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _now: SystemTime,
    ) -> Result<Result<VirtualKey, KeyDenial>, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl ModelBudgetCore for Unported {
    async fn check_key_and_user<H: ModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    async fn check_user<H: ModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    async fn check_team_member<H: TeamMemberModelBudgetHost>(
        &self,
        _host: &H,
    ) -> Result<ModelBudgetResult, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl AuthPlanner for Unported {
    fn plan(&self, _facts: &PlanFacts, _settings: &PlanSettings) -> Plan {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    fn after_custom_auth(
        &self,
        _facts: &PlanFacts,
        _settings: &PlanSettings,
        _issued_key: bool,
    ) -> CredentialPlan {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    fn after_jwt(
        &self,
        _facts: &PlanFacts,
        _settings: &PlanSettings,
        _issued_key: bool,
    ) -> KeyPlan {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    fn oauth2_introspection_applies(&self, _facts: &PlanFacts, _settings: &PlanSettings) -> bool {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl AnonymousCore for Unported {
    fn authenticate(&self, _reason: AnonymousReason) -> Anonymous {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl JwtCore for Unported {
    async fn authenticate<H: JwtHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
        _settings: &JwtSettings,
        _now: SystemTime,
    ) -> Result<Result<Identity, AuthDenial>, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl OAuth2Core for Unported {
    async fn introspect<H: OAuth2IntrospectionHost>(
        &self,
        _host: &H,
        _token: &str,
        _settings: &OAuth2IntrospectionSettings,
    ) -> Result<Result<OAuth2Identity, OAuth2Denial>, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    fn proxy<H: OAuth2ProxyHost>(
        &self,
        _host: &H,
        _settings: &OAuth2ProxySettings,
    ) -> Result<OAuth2ProxyIdentity, OAuth2ProxyDenial> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl CustomAuthCore for Unported {
    async fn authenticate<H: CustomAuthHost>(
        &self,
        _host: &H,
        _facts: &RequestFacts,
    ) -> Result<CustomAuthResult, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}

impl AuthorizeCore for Unported {
    async fn authorize<H: AuthorizeHost>(
        &self,
        _host: &H,
        _identity: &KeyIdentity,
        _facts: &RequestFacts,
        _settings: &AuthorizeSettings,
    ) -> Result<Outcome, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }

    async fn check_request<H: RequestCheckHost>(
        &self,
        _host: &H,
        _facts: &CheckFacts,
        _settings: &CheckSettings,
    ) -> Result<RequestCheck, H::Error> {
        unimplemented!("{RUNS_IN_PYTHON}")
    }
}
