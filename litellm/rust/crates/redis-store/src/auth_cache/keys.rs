use litellm_python_compat::{Value, json};

/// One object the auth cache holds. `key` writes the bytes `main`'s Python writes for it. Some of
/// those spellings are ambiguous (a bare user id, `{team_id}_{user_id}`, a JWT claim value holding
/// `:`), so a key is never parsed back: writers name the object, and evictions carry wire keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum AuthObject {
    /// A virtual key, cached under its bare `hash_token` digest.
    Key(String),
    User(String),
    Team(String),
    TeamAlias(String),
    /// The admission check's `LiteLLM_TeamMembership` row.
    TeamMembership {
        team_id: String,
        user_id: String,
    },
    /// The budget reservation's copy of the same row, under a key of its own on `main`.
    TeamMembershipReservation {
        user_id: String,
        team_id: String,
    },
    Org(String),
    OrgWithBudget(String),
    OrgWithBudgetLastKnown(String),
    OrgAlias(String),
    Project(String),
    EndUser(String),
    EndUserRestrictedRegistry,
    EndUserValidation(String),
    CustomerMember(String),
    CustomerOrgAlias(String),
    /// The one customer app definition every app shares.
    CustomerAppDefinition,
    CustomerApp {
        resource_type: String,
        resource_id: String,
    },
    CustomerAppAssociation {
        scope: String,
        app_id: String,
        customer_id: String,
    },
    CustomerAppBudget(String),
    CustomerAppKeyResource(String),
    CustomerAppSharedProfiles,
    DefaultEndUserBudget(String),
    TeamMemberDefaultBudget(String),
    Tag(String),
    TagRegistry,
    ModelAccessGroup(String),
    ModelAccessGroupRegistry,
    AccessGroup(String),
    AccessGroupRuleName(String),
    ObjectPermission(String),
    UserObjectPermission(String),
    AgentObjectPermission(String),
    ManagedVectorStore(String),
    JwtKeyMapping {
        issuer: Option<String>,
        claim_name: String,
        claim_value: String,
        strict: bool,
    },
    AgentIdentity {
        tenant_id: String,
        client_id: String,
    },
    GlobalProxySpend,
}

impl AuthObject {
    pub fn key(&self) -> String {
        match self {
            Self::Key(hashed_token) => hashed_token.clone(),
            Self::User(user_id) => user_id.clone(),
            Self::Team(id) => format!("team_id:{id}"),
            Self::TeamAlias(alias) => format!("team_alias:{alias}"),
            Self::TeamMembership { team_id, user_id } => format!("{team_id}_{user_id}"),
            Self::TeamMembershipReservation { user_id, team_id } => {
                format!("team_membership:{user_id}:{team_id}")
            }
            Self::Org(id) => format!("org_id:{id}"),
            Self::OrgWithBudget(id) => format!("org_id:{id}:with_budget"),
            Self::OrgWithBudgetLastKnown(id) => format!("org_id:{id}:with_budget:last_known"),
            Self::OrgAlias(alias) => format!("org_alias:{alias}"),
            Self::Project(id) => format!("project_id:{id}"),
            Self::EndUser(id) => format!("end_user_id:{id}"),
            Self::EndUserRestrictedRegistry => "end_user_restricted_registry".to_owned(),
            Self::EndUserValidation(id) => format!("end_user_validation:{id}"),
            Self::CustomerMember(id) => format!("customer_member_id:{id}"),
            Self::CustomerOrgAlias(alias) => format!("customer_org_alias:{alias}"),
            Self::CustomerAppDefinition => "customer_app_definition:global".to_owned(),
            Self::CustomerApp {
                resource_type,
                resource_id,
            } => format!(
                "customer_app:{}",
                json_strings(&[resource_type, resource_id])
            ),
            Self::CustomerAppAssociation {
                scope,
                app_id,
                customer_id,
            } => format!(
                "customer_app_association:{}",
                json_strings(&[scope, app_id, customer_id])
            ),
            Self::CustomerAppBudget(id) => format!("customer_app_budget:{id}"),
            Self::CustomerAppKeyResource(hashed_token) => {
                format!("customer_app_key_resource:{hashed_token}")
            }
            Self::CustomerAppSharedProfiles => "customer_app_shared_profiles".to_owned(),
            Self::DefaultEndUserBudget(id) => format!("default_end_user_budget:{id}"),
            Self::TeamMemberDefaultBudget(id) => format!("team_member_default_budget:{id}"),
            Self::Tag(name) => format!("tag:{name}"),
            Self::TagRegistry => "tag_registry".to_owned(),
            Self::ModelAccessGroup(name) => format!("model_access_group:{name}"),
            Self::ModelAccessGroupRegistry => "model_access_group_registry".to_owned(),
            Self::AccessGroup(id) => format!("access_group_id:{id}"),
            Self::AccessGroupRuleName(name) => format!("access_group_rule_name:{name}"),
            Self::ObjectPermission(id) => format!("object_permission_id:{id}"),
            Self::UserObjectPermission(user_id) => format!("user_object_permission_id:{user_id}"),
            Self::AgentObjectPermission(agent_id) => {
                format!("agent_object_permission_id:{agent_id}")
            }
            Self::ManagedVectorStore(id) => format!("managed_vector_store_id:{id}"),
            Self::JwtKeyMapping {
                issuer,
                claim_name,
                claim_value,
                strict: true,
            } => format!(
                "jwt_key_mapping_strict:{}",
                json_strings(&[
                    issuer.as_deref().unwrap_or_default(),
                    claim_name,
                    claim_value
                ])
            ),
            Self::JwtKeyMapping {
                issuer,
                claim_name,
                claim_value,
                strict: false,
            } => match issuer.as_deref().filter(|issuer| !issuer.is_empty()) {
                None => format!("jwt_key_mapping:{claim_name}:{claim_value}"),
                Some(issuer) => format!("jwt_key_mapping:{issuer}:{claim_name}:{claim_value}"),
            },
            Self::AgentIdentity {
                tenant_id,
                client_id,
            } => format!("agent_identity:{}", json_strings(&[tenant_id, client_id])),
            Self::GlobalProxySpend => "default_user_id:spend".to_owned(),
        }
    }

    /// User-key objects live in a memory partition of their own, so churn in management objects
    /// cannot evict them.
    pub fn is_key_object(&self) -> bool {
        matches!(self, Self::Key(_))
    }
}

/// `json.dumps(tuple(items))`.
fn json_strings(items: &[&str]) -> String {
    let list = Value::List(items.iter().copied().map(Value::from).collect());
    json::dumps(&list).expect("json.dumps never refuses a list of strings")
}
