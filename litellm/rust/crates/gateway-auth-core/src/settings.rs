use litellm_config::{GeneralSettings, JwtAuthSettings, RolePermissions};

use crate::{
    MasterKey,
    authn::{jwt::JwtSettings, key::KeySettings},
    authz::RoutePolicy,
    identity::jwt::TeamSettings,
    plan,
    routes::{RouteGroup, RouteGroups},
};

/// What the host knows that `general_settings` does not say.
pub struct HostFacts {
    /// `proxy_server.master_key`, resolved from `general_settings.master_key` or `LITELLM_MASTER_KEY`
    pub master_key: Option<MasterKey>,
    /// `premium_user`, a valid `LITELLM_LICENSE`
    pub premium: bool,
    /// `prisma_client is not None`
    pub has_db: bool,
    /// `EXPERIMENTAL_UI_LOGIN` is not false
    pub ui_login: bool,
    /// `litellm.max_budget > 0`
    pub proxy_budget: bool,
    /// `litellm.require_auth_for_metrics_endpoint is not False`
    pub metrics_require_auth: bool,
    /// `LiteLLMRoutes`
    pub route_groups: RouteGroups,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub plan: plan::Settings,
    pub key: KeySettings,
    pub jwt: JwtSettings,
}

fn jwt_settings(
    jwt: &JwtAuthSettings,
    role_permissions: Option<&[RolePermissions]>,
    env: &dyn Fn(&str) -> Option<String>,
) -> JwtSettings {
    JwtSettings {
        config: jwt.clone(),
        public_key_urls: env("JWT_PUBLIC_KEY_URL")
            .map(|urls| urls.split(',').map(|url| url.trim().to_owned()).collect())
            .unwrap_or_default(),
        audience: env("JWT_AUDIENCE"),
        issuer: env("JWT_ISSUER"),
        leeway_seconds: 0,
        role_permissions: role_permissions.unwrap_or_default().into(),
        ..JwtSettings::default()
    }
}

impl Settings {
    pub fn new(
        general: &GeneralSettings,
        facts: HostFacts,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        let has_master_key = facts.master_key.is_some();
        Self {
            plan: plan::Settings::new(
                general,
                facts.premium,
                facts.metrics_require_auth,
                &facts.route_groups,
            ),
            key: KeySettings {
                master_key: facts.master_key,
                has_db: facts.has_db,
                allow_user_auth: general.allow_user_auth == Some(true),
                ui_login: facts.ui_login,
                jwt_auth: general.enable_jwt_auth == Some(true),
                proxy_budget: facts.proxy_budget,
                master_key_only_routes: facts
                    .route_groups
                    .routes(RouteGroup::MasterKeyOnly)
                    .iter()
                    .cloned()
                    .collect(),
                min_custom_key_length: env("MINIMUM_CUSTOM_KEY_LENGTH")
                    .and_then(|length| length.trim().parse().ok())
                    .unwrap_or(16),
            },
            jwt: JwtSettings {
                has_db: facts.has_db,
                has_master_key,
                enforce_rbac: general.enforce_rbac == Some(true),
                route_groups: facts.route_groups.clone(),
                ..jwt_settings(
                    &general.litellm_jwtauth.clone().unwrap_or_default(),
                    general.role_permissions.as_deref(),
                    env,
                )
            },
        }
    }
}

impl Settings {
    pub(crate) fn policy(&self) -> RoutePolicy<'_> {
        RoutePolicy::new(&self.key, &self.jwt)
    }
}

impl From<&JwtAuthSettings> for TeamSettings {
    fn from(jwt: &JwtAuthSettings) -> Self {
        Self {
            enforce_team_based_model_access: jwt.enforce_team_based_model_access,
            fallback_to_db_teams: jwt.fallback_to_db_teams,
            team_claim_fallback: jwt.team_claim_fallback,
        }
    }
}
