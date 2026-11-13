use litellm_config::{ClaimPath, LitellmRole, RbacRole};
use litellm_fnmatch::fnmatchcase;
use serde_json::{Map, Value};

use super::{ClaimDenial, JwtSettings, verify::Claims};
use crate::value::{id_text, is_separator, is_set, kind};

/// Claims the verifier writes after normalizing a token. Never trusted from the token itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::EnumString)]
pub(super) enum InternalClaim {
    #[strum(serialize = "_litellm_jwt_issuer")]
    Issuer,
    #[strum(serialize = "_litellm_user_id")]
    UserId,
    #[strum(serialize = "_litellm_user_email")]
    UserEmail,
    #[strum(serialize = "_litellm_team_id")]
    TeamId,
    #[strum(serialize = "_litellm_team_ids")]
    TeamIds,
    #[strum(serialize = "_litellm_org_id")]
    OrgId,
    #[strum(serialize = "_litellm_end_user_id")]
    EndUserId,
}

/// `get_nested_value(data, key_path)` with no default. Only objects are walked into.
pub fn nested_value<'a>(data: &'a Map<String, Value>, path: &ClaimPath) -> Option<&'a Value> {
    let (first, rest) = path.keys().split_first()?;
    rest.iter().try_fold(data.get(first)?, |current, key| {
        current.as_object()?.get(key)
    })
}

fn configured<'a>(claims: &'a Claims, field: Option<&ClaimPath>) -> Option<&'a Value> {
    nested_value(&claims.0, field?)
}

/// The trusted normalized claim when the token has one, even `null`, else the configured field.
fn trusted_or_configured<'a>(
    claims: &'a Claims,
    settings: &JwtSettings,
    claim: InternalClaim,
    field: Option<&ClaimPath>,
) -> Option<&'a Value> {
    trusted_claim(claims, settings, claim).or_else(|| configured(claims, field))
}

/// The string items of a role claim. A claim that is not an array names no roles.
fn role_strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// `_is_trusted_issuer_normalized_token`
pub fn is_trusted_issuer_normalized(claims: &Claims, settings: &JwtSettings) -> bool {
    let Some(issuer) = claims
        .0
        .get(<&str>::from(InternalClaim::Issuer))
        .and_then(Value::as_str)
        .filter(|issuer| !issuer.is_empty())
    else {
        return false;
    };
    settings
        .config
        .issuers
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|config| config.issuer == issuer)
}

/// `_has_trusted_issuer_normalized_claim`, returning the claim it found.
fn trusted_claim<'a>(
    claims: &'a Claims,
    settings: &JwtSettings,
    claim: InternalClaim,
) -> Option<&'a Value> {
    claims
        .0
        .get(<&str>::from(claim))
        .filter(|_| is_trusted_issuer_normalized(claims, settings))
}

/// `get_team_ids_from_jwt`
pub fn team_ids(claims: &Claims, settings: &JwtSettings) -> Vec<String> {
    let items = |value: &Value| match value {
        Value::Array(items) => items.iter().filter_map(id_text).collect(),
        _ => Vec::new(),
    };
    if let Some(value) = trusted_claim(claims, settings, InternalClaim::TeamIds) {
        return match value {
            Value::String(team_id) => vec![team_id.clone()],
            other => items(other),
        };
    }
    configured(claims, settings.config.team_ids_jwt_field.as_ref())
        .map(items)
        .unwrap_or_default()
}

/// `get_all_jwt_team_ids`: the plural claim's ids, then the singular claim's ids not already seen.
pub fn all_team_ids(claims: &Claims, settings: &JwtSettings) -> Vec<String> {
    let plural = team_ids(claims, settings);
    let singular = trusted_or_configured(
        claims,
        settings,
        InternalClaim::TeamId,
        settings.config.team_id_jwt_field.as_ref(),
    );
    let extra: Vec<String> = match singular {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(id_text)
            .filter(|id| !id.is_empty())
            .collect(),
        Some(value) if is_set(value) => id_text(value).into_iter().collect(),
        Some(_) => Vec::new(),
    };
    extra.into_iter().fold(plural, |mut ids, id| {
        if !ids.contains(&id) {
            ids.push(id);
        }
        ids
    })
}

/// `get_end_user_id`
pub fn end_user_id(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    let field = settings.config.end_user_id_jwt_field.as_ref();
    trusted_or_configured(claims, settings, InternalClaim::EndUserId, field).and_then(id_text)
}

/// `is_required_team_id`
pub fn is_required_team_id(settings: &JwtSettings) -> bool {
    settings.config.team_id_jwt_field.is_some() || settings.config.team_alias_jwt_field.is_some()
}

/// `is_enforced_email_domain`
pub fn is_enforced_email_domain(settings: &JwtSettings) -> bool {
    settings.config.user_allowed_email_domain.is_some()
}

/// `get_team_id`: a list claim gives its first item, and a missing claim gives `team_id_default`.
pub fn team_id(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    let first_or_value = |value: &Value| match value {
        Value::Array(items) => items.first().and_then(id_text),
        other => id_text(other),
    };
    let field = settings.config.team_id_jwt_field.as_ref();
    match trusted_or_configured(claims, settings, InternalClaim::TeamId, field) {
        None => settings.config.team_id_default.clone(),
        Some(value) => first_or_value(value),
    }
}

/// `get_team_alias`
pub fn team_alias(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    configured(claims, settings.config.team_alias_jwt_field.as_ref()).and_then(id_text)
}

/// `is_upsert_user_id`
pub fn is_upsert_user_id(settings: &JwtSettings, valid_user_email: Option<bool>) -> bool {
    valid_user_email != Some(false) && settings.config.user_id_upsert
}

/// `get_user_id(token, default_value)`. A non-string claim gives the default when one is passed.
pub fn user_id(claims: &Claims, settings: &JwtSettings, default: Option<&str>) -> Option<String> {
    if let Some(value) = trusted_claim(claims, settings, InternalClaim::UserId) {
        return id_text(value);
    }
    let value = configured(claims, settings.config.user_id_jwt_field.as_ref());
    match default {
        None => value.and_then(id_text),
        Some(default) => Some(value.and_then(Value::as_str).unwrap_or(default).to_owned()),
    }
}

/// `get_user_roles(token, None)`
pub fn user_roles(claims: &Claims, settings: &JwtSettings) -> Option<Vec<String>> {
    configured(claims, settings.config.user_roles_jwt_field.as_ref())
        .filter(|value| !value.is_null())
        .map(role_strings)
}

/// `get_jwt_role(token, None)`
pub fn jwt_roles(claims: &Claims, settings: &JwtSettings) -> Option<Vec<String>> {
    configured(claims, settings.config.roles_jwt_field.as_ref())
        .filter(|value| !value.is_null())
        .map(role_strings)
}

/// `map_jwt_role_to_litellm_role`: the first mapping whose glob matches any role in the claim's list.
pub fn litellm_role(claims: &Claims, settings: &JwtSettings) -> Option<LitellmRole> {
    let mappings = settings
        .config
        .jwt_litellm_role_map
        .as_deref()
        .filter(|mappings| !mappings.is_empty())?;
    let roles = role_strings(configured(
        claims,
        settings.config.roles_jwt_field.as_ref(),
    )?);
    mappings
        .iter()
        .find(|mapping| {
            roles
                .iter()
                .any(|role| fnmatchcase(role, &mapping.jwt_role))
        })
        .map(|mapping| mapping.litellm_role)
}

/// `is_allowed_user_role`
pub fn is_allowed_user_role(user_roles: Option<&[String]>, settings: &JwtSettings) -> bool {
    match (user_roles, settings.config.user_allowed_roles.as_deref()) {
        (Some(roles), Some(allowed)) => roles.iter().any(|role| allowed.contains(role)),
        _ => false,
    }
}

/// `get_user_email`
pub fn user_email(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    let field = settings.config.user_email_jwt_field.as_ref();
    trusted_or_configured(claims, settings, InternalClaim::UserEmail, field).and_then(id_text)
}

/// `get_object_id`
pub fn object_id(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    configured(claims, settings.config.object_id_jwt_field.as_ref()).and_then(id_text)
}

/// `get_agent_claim`: only a non-empty string.
pub fn agent_claim(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    configured(claims, settings.config.agent_id_jwt_field.as_ref())?
        .as_str()
        .filter(|claim| !claim.is_empty())
        .map(str::to_owned)
}

/// `get_org_id`
pub fn org_id(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    let field = settings.config.org_id_jwt_field.as_ref();
    trusted_or_configured(claims, settings, InternalClaim::OrgId, field).and_then(id_text)
}

/// `get_org_alias`
pub fn org_alias(claims: &Claims, settings: &JwtSettings) -> Option<String> {
    configured(claims, settings.config.org_alias_jwt_field.as_ref()).and_then(id_text)
}

/// `get_scopes`: a space-separated string or a list in the `scope` claim.
pub fn scopes(claims: &Claims) -> Result<Vec<String>, ClaimDenial> {
    match claims.0.get("scope") {
        None => Ok(Vec::new()),
        Some(Value::String(scope)) => Ok(scope
            .split(is_separator)
            .filter(|scope| !scope.is_empty())
            .map(str::to_owned)
            .collect()),
        Some(Value::Array(items)) => Ok(items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()),
        Some(other) => Err(ClaimDenial::UnmappedScopeType { kind: kind(other) }),
    }
}

/// `is_admin`
pub fn is_admin(scopes: &[String], settings: &JwtSettings) -> bool {
    scopes.contains(&settings.config.admin_jwt_scope)
}

/// `_rbac_role_from_role_mapping`
fn role_from_mapping(claims: &Claims, settings: &JwtSettings) -> Option<RbacRole> {
    let mappings = settings.config.role_mappings.as_deref()?;
    let claim = configured(claims, settings.config.roles_jwt_field.as_ref())
        .filter(|value| is_set(value))?;
    let roles = role_strings(claim);
    mappings
        .iter()
        .find(|mapping| roles.contains(&mapping.role))
        .map(|mapping| mapping.internal_role)
}

/// `get_rbac_role`: admin scope, then a team id, then a user id or allowed user role, then role mappings.
pub fn rbac_role(claims: &Claims, settings: &JwtSettings) -> Result<Option<RbacRole>, ClaimDenial> {
    if is_admin(&scopes(claims)?, settings) {
        return Ok(Some(RbacRole::ProxyAdmin));
    }
    if team_id(claims, settings).is_some() {
        return Ok(Some(RbacRole::Team));
    }
    if user_id(claims, settings, None).is_some()
        || is_allowed_user_role(user_roles(claims, settings).as_deref(), settings)
    {
        return Ok(Some(RbacRole::InternalUser));
    }
    Ok(role_from_mapping(claims, settings))
}

/// `is_allowed_domain`: the part after the last `@` must be `user_allowed_email_domain`.
pub fn is_allowed_domain(user_email: &str, settings: &JwtSettings) -> bool {
    match settings.config.user_allowed_email_domain.as_deref() {
        None => true,
        Some(domain) => user_email.rsplit('@').next() == Some(domain),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UserInfo {
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    /// `None` when no email domain is enforced.
    pub valid_user_email: Option<bool>,
}

/// `JWTAuthManager.get_user_info`: the user id falls back to the email.
pub fn user_info(claims: &Claims, settings: &JwtSettings) -> UserInfo {
    let user_email = user_email(claims, settings);
    let valid_user_email = is_enforced_email_domain(settings).then(|| {
        user_email
            .as_deref()
            .is_some_and(|email| is_allowed_domain(email, settings))
    });
    UserInfo {
        user_id: user_id(claims, settings, user_email.as_deref()),
        user_email,
        valid_user_email,
    }
}

/// The readers that cannot fail, for callers that want them together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MappedClaims {
    pub user: UserInfo,
    pub team_id: Option<String>,
    pub team_ids: Vec<String>,
    pub all_team_ids: Vec<String>,
    pub team_alias: Option<String>,
    pub org_id: Option<String>,
    pub org_alias: Option<String>,
    pub end_user_id: Option<String>,
    pub object_id: Option<String>,
    pub agent_claim: Option<String>,
}

pub fn claims(claims: &Claims, settings: &JwtSettings) -> MappedClaims {
    MappedClaims {
        user: user_info(claims, settings),
        team_id: team_id(claims, settings),
        team_ids: team_ids(claims, settings),
        all_team_ids: all_team_ids(claims, settings),
        team_alias: team_alias(claims, settings),
        org_id: org_id(claims, settings),
        org_alias: org_alias(claims, settings),
        end_user_id: end_user_id(claims, settings),
        object_id: object_id(claims, settings),
        agent_claim: agent_claim(claims, settings),
    }
}

#[cfg(test)]
mod tests {
    use litellm_config::{JwtAuthSettings, JwtIssuerConfig, JwtLitellmRoleMap, RoleMapping};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn claims(value: Value) -> Claims {
        match value {
            Value::Object(map) => Claims(map),
            other => panic!("claims must be an object, got {other}"),
        }
    }

    fn settings(config: JwtAuthSettings) -> JwtSettings {
        JwtSettings {
            config,
            ..JwtSettings::default()
        }
    }

    fn with_issuer(config: JwtAuthSettings) -> JwtSettings {
        settings(JwtAuthSettings {
            issuers: Some(Box::new([JwtIssuerConfig {
                issuer: "https://idp".to_owned(),
                ..JwtIssuerConfig::default()
            }])),
            ..config
        })
    }

    #[rstest]
    #[case::top_level(json!({"sub": "u"}), "sub", Some(json!("u")))]
    #[case::nested(json!({"a": {"b": {"c": 1}}}), "a.b.c", Some(json!(1)))]
    #[case::escaped_dot(json!({"kubernetes.io": {"namespace": "ns"}}), "kubernetes\\.io.namespace", Some(json!("ns")))]
    #[case::metadata_prefix_dropped(json!({"team": "t"}), "metadata.team", Some(json!("t")))]
    #[case::missing(json!({"a": {}}), "a.b", None)]
    #[case::through_list(json!({"a": [{"b": 1}]}), "a.0.b", None)]
    #[case::through_string(json!({"a": "text"}), "a.b", None)]
    #[case::empty_path(json!({"": 1}), "", None)]
    #[case::metadata_prefix_alone_names_empty_key(json!({"": 1}), "metadata.", Some(json!(1)))]
    #[case::escaped_dots_in_a_row(json!({"a..b": 1}), "a\\.\\.b", Some(json!(1)))]
    #[case::nul_is_not_a_dot(json!({"a\u{0}b": 1, "a.b": 2}), "a\u{0}b", Some(json!(1)))]
    #[case::null_kept(json!({"a": null}), "a", Some(Value::Null))]
    fn nested_value_walks_objects(
        #[case] token: Value,
        #[case] path: &str,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(
            nested_value(&claims(token).0, &ClaimPath::from(path)).cloned(),
            expected
        );
    }

    #[rstest]
    #[case::string_split_on_whitespace(json!({"scope": " a\tb  c "}), Ok(vec!["a", "b", "c"]))]
    #[case::list(json!({"scope": ["a", "b"]}), Ok(vec!["a", "b"]))]
    #[case::missing(json!({}), Ok(vec![]))]
    #[case::int(json!({"scope": 1}), Err(ClaimDenial::UnmappedScopeType { kind: "number" }))]
    #[case::null(json!({"scope": null}), Err(ClaimDenial::UnmappedScopeType { kind: "null" }))]
    fn scopes_read_string_or_list(
        #[case] token: Value,
        #[case] expected: Result<Vec<&str>, ClaimDenial>,
    ) {
        let expected =
            expected.map(|scopes| scopes.into_iter().map(str::to_owned).collect::<Vec<_>>());
        assert_eq!(scopes(&claims(token)), expected);
    }

    fn rbac_config() -> JwtAuthSettings {
        JwtAuthSettings {
            team_id_jwt_field: Some("team".into()),
            user_id_jwt_field: Some("sub".into()),
            user_roles_jwt_field: Some("user_roles".into()),
            user_allowed_roles: Some(Box::new(["member".to_owned()])),
            roles_jwt_field: Some("roles".into()),
            role_mappings: Some(Box::new([RoleMapping {
                role: "ops".to_owned(),
                internal_role: RbacRole::ProxyAdmin,
            }])),
            ..JwtAuthSettings::default()
        }
    }

    #[rstest]
    #[case::admin_scope_wins(json!({"scope": "litellm_proxy_admin", "team": "t", "sub": "u"}), Some(RbacRole::ProxyAdmin))]
    #[case::team_before_user(json!({"team": "t", "sub": "u"}), Some(RbacRole::Team))]
    #[case::user_id(json!({"sub": "u"}), Some(RbacRole::InternalUser))]
    #[case::allowed_user_role(json!({"user_roles": ["member"]}), Some(RbacRole::InternalUser))]
    #[case::other_user_role(json!({"user_roles": ["guest"]}), None)]
    #[case::role_mapping(json!({"roles": ["ops"]}), Some(RbacRole::ProxyAdmin))]
    #[case::role_mapping_from_string_is_ignored(json!({"roles": "ops"}), None)]
    #[case::role_mapping_from_object_is_ignored(json!({"roles": {"ops": true}}), None)]
    #[case::nothing(json!({}), None)]
    fn rbac_role_in_precedence_order(#[case] token: Value, #[case] expected: Option<RbacRole>) {
        assert_eq!(
            rbac_role(&claims(token), &settings(rbac_config())),
            Ok(expected)
        );
    }

    #[test]
    fn rbac_role_fails_on_unmapped_scope() {
        assert_eq!(
            rbac_role(
                &claims(json!({"scope": {"a": 1}, "team": "t"})),
                &settings(rbac_config())
            ),
            Err(ClaimDenial::UnmappedScopeType { kind: "object" })
        );
    }

    fn team_config() -> JwtAuthSettings {
        JwtAuthSettings {
            team_id_jwt_field: Some("groups".into()),
            team_id_default: Some("default-team".to_owned()),
            ..JwtAuthSettings::default()
        }
    }

    #[rstest]
    #[case::string(json!({"groups": "t1"}), Some("t1"))]
    #[case::list_first(json!({"groups": ["t1", "t2"]}), Some("t1"))]
    #[case::empty_list(json!({"groups": []}), None)]
    #[case::missing_uses_default(json!({}), Some("default-team"))]
    #[case::null_is_not_missing(json!({"groups": null}), None)]
    #[case::integer_as_decimal(json!({"groups": 7}), Some("7"))]
    #[case::float_is_no_id(json!({"groups": 7.5}), None)]
    #[case::bool_is_no_id(json!({"groups": true}), None)]
    fn team_id_reads_singular_claim(#[case] token: Value, #[case] expected: Option<&str>) {
        assert_eq!(
            team_id(&claims(token), &settings(team_config())),
            expected.map(str::to_owned)
        );
    }

    #[test]
    fn team_id_without_field_is_default() {
        let config = JwtAuthSettings {
            team_id_jwt_field: None,
            ..team_config()
        };
        assert_eq!(
            team_id(&claims(json!({"groups": "t1"})), &settings(config)),
            Some("default-team".to_owned())
        );
    }

    #[rstest]
    #[case::trusted_list_first(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_team_id": ["n1", "n2"], "groups": "g"}), Some("n1"))]
    #[case::trusted_null(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_team_id": null, "groups": "g"}), None)]
    #[case::untrusted_issuer_ignored(json!({"_litellm_jwt_issuer": "https://other", "_litellm_team_id": "n1", "groups": "g"}), Some("g"))]
    #[case::trusted_without_claim(json!({"_litellm_jwt_issuer": "https://idp", "groups": "g"}), Some("g"))]
    fn team_id_prefers_trusted_normalized_claim(
        #[case] token: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            team_id(&claims(token), &with_issuer(team_config())),
            expected.map(str::to_owned)
        );
    }

    fn team_ids_config() -> JwtAuthSettings {
        JwtAuthSettings {
            team_ids_jwt_field: Some("teams".into()),
            team_id_jwt_field: Some("team".into()),
            team_id_default: Some("ignored".to_owned()),
            ..JwtAuthSettings::default()
        }
    }

    #[rstest]
    #[case::list(json!({"teams": ["a", "b"]}), vec!["a", "b"])]
    #[case::string_is_not_a_list(json!({"teams": "a"}), vec![])]
    #[case::trusted_string(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_team_ids": "n", "teams": ["a"]}), vec!["n"])]
    #[case::trusted_other_type_blocks_fallback(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_team_ids": 3, "teams": ["a"]}), vec![])]
    fn team_ids_read_plural_claim(#[case] token: Value, #[case] expected: Vec<&str>) {
        assert_eq!(
            team_ids(&claims(token), &with_issuer(team_ids_config())),
            expected
        );
    }

    #[rstest]
    #[case::plural_then_singular(json!({"teams": ["a"], "team": "b"}), vec!["a", "b"])]
    #[case::singular_duplicate_dropped(json!({"teams": ["a", "b"], "team": "a"}), vec!["a", "b"])]
    #[case::plural_duplicates_kept(json!({"teams": ["a", "a"]}), vec!["a", "a"])]
    #[case::singular_list(json!({"teams": ["a"], "team": ["b", null, "", "a", 5]}), vec!["a", "b", "5"])]
    #[case::falsy_singular_skipped(json!({"team": 0}), vec![])]
    #[case::default_not_consulted(json!({}), vec![])]
    fn all_team_ids_merge_both_claims(#[case] token: Value, #[case] expected: Vec<&str>) {
        assert_eq!(
            all_team_ids(&claims(token), &settings(team_ids_config())),
            expected
        );
    }

    fn user_config() -> JwtAuthSettings {
        JwtAuthSettings {
            user_id_jwt_field: Some("sub".into()),
            user_email_jwt_field: Some("email".into()),
            user_allowed_email_domain: Some("corp.com".to_owned()),
            ..JwtAuthSettings::default()
        }
    }

    #[rstest]
    #[case::allowed(json!({"sub": "u", "email": "a@corp.com"}), Some("u"), Some("a@corp.com"), Some(true))]
    #[case::other_domain(json!({"sub": "u", "email": "a@x.com"}), Some("u"), Some("a@x.com"), Some(false))]
    #[case::last_at_decides(json!({"email": "a@corp.com@x.com"}), Some("a@corp.com@x.com"), Some("a@corp.com@x.com"), Some(false))]
    #[case::no_email(json!({"sub": "u"}), Some("u"), None, Some(false))]
    #[case::user_id_falls_back_to_email(json!({"email": "a@corp.com"}), Some("a@corp.com"), Some("a@corp.com"), Some(true))]
    #[case::non_string_user_id_falls_back_to_email(json!({"sub": 5, "email": "a@corp.com"}), Some("a@corp.com"), Some("a@corp.com"), Some(true))]
    fn user_info_checks_email_domain(
        #[case] token: Value,
        #[case] user_id: Option<&str>,
        #[case] user_email: Option<&str>,
        #[case] valid_user_email: Option<bool>,
    ) {
        assert_eq!(
            user_info(&claims(token), &settings(user_config())),
            UserInfo {
                user_id: user_id.map(str::to_owned),
                user_email: user_email.map(str::to_owned),
                valid_user_email,
            }
        );
    }

    #[test]
    fn user_info_without_enforced_domain_leaves_validity_unset() {
        let config = JwtAuthSettings {
            user_allowed_email_domain: None,
            ..user_config()
        };
        assert_eq!(
            user_info(&claims(json!({"email": "a@x.com"})), &settings(config)).valid_user_email,
            None
        );
    }

    #[rstest]
    #[case::configured_field(json!({"sub": 7}), None, Some("7"))]
    #[case::trusted_claim_ignores_default(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_user_id": null, "sub": "u"}), Some("d"), None)]
    #[case::missing_uses_default(json!({}), Some("d"), Some("d"))]
    fn user_id_honors_default(
        #[case] token: Value,
        #[case] default: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            user_id(&claims(token), &with_issuer(user_config()), default),
            expected.map(str::to_owned)
        );
    }

    #[test]
    fn user_id_without_field_is_default() {
        assert_eq!(
            user_id(
                &claims(json!({"sub": "u"})),
                &settings(JwtAuthSettings::default()),
                Some("d")
            ),
            Some("d".to_owned())
        );
    }

    #[rstest]
    #[case::end_user(json!({"eu": "e", "_litellm_jwt_issuer": "https://idp"}), end_user_id, Some("e"))]
    #[case::end_user_trusted(json!({"eu": "e", "_litellm_jwt_issuer": "https://idp", "_litellm_end_user_id": "n"}), end_user_id, Some("n"))]
    #[case::org(json!({"org": {"id": "o"}}), org_id, Some("o"))]
    #[case::org_trusted(json!({"org": {"id": "o"}, "_litellm_jwt_issuer": "https://idp", "_litellm_org_id": "n"}), org_id, Some("n"))]
    #[case::org_alias(json!({"org_name": "acme"}), org_alias, Some("acme"))]
    #[case::team_alias(json!({"team_name": "t"}), team_alias, Some("t"))]
    #[case::email_trusted(json!({"_litellm_jwt_issuer": "https://idp", "_litellm_user_email": "n@x"}), user_email, Some("n@x"))]
    #[case::object_id(json!({"oid": "obj"}), object_id, Some("obj"))]
    #[case::agent(json!({"agent": "a1"}), agent_claim, Some("a1"))]
    #[case::agent_empty(json!({"agent": ""}), agent_claim, None)]
    #[case::agent_not_string(json!({"agent": 1}), agent_claim, None)]
    fn scalar_readers(
        #[case] token: Value,
        #[case] reader: fn(&Claims, &JwtSettings) -> Option<String>,
        #[case] expected: Option<&str>,
    ) {
        let config = JwtAuthSettings {
            end_user_id_jwt_field: Some("eu".into()),
            org_id_jwt_field: Some("org.id".into()),
            org_alias_jwt_field: Some("org_name".into()),
            team_alias_jwt_field: Some("team_name".into()),
            object_id_jwt_field: Some("oid".into()),
            agent_id_jwt_field: Some("agent".into()),
            ..JwtAuthSettings::default()
        };
        assert_eq!(
            reader(&claims(token), &with_issuer(config)),
            expected.map(str::to_owned)
        );
    }

    #[rstest]
    #[case::end_user(end_user_id)]
    #[case::org(org_id)]
    #[case::org_alias(org_alias)]
    #[case::team_alias(team_alias)]
    #[case::email(user_email)]
    #[case::object_id(object_id)]
    #[case::agent(agent_claim)]
    fn scalar_readers_need_a_field(#[case] reader: fn(&Claims, &JwtSettings) -> Option<String>) {
        let token = json!({"sub": "u", "email": "e", "eu": "e", "org": "o", "agent": "a"});
        assert_eq!(
            reader(&claims(token), &settings(JwtAuthSettings::default())),
            None
        );
    }

    fn role_map_config(jwt_role: &str) -> JwtAuthSettings {
        JwtAuthSettings {
            roles_jwt_field: Some("roles".into()),
            jwt_litellm_role_map: Some(Box::new([
                JwtLitellmRoleMap {
                    jwt_role: jwt_role.to_owned(),
                    litellm_role: LitellmRole::ProxyAdmin,
                },
                JwtLitellmRoleMap {
                    jwt_role: "*".to_owned(),
                    litellm_role: LitellmRole::InternalUser,
                },
            ])),
            ..JwtAuthSettings::default()
        }
    }

    #[rstest]
    #[case::first_mapping_wins(json!({"roles": ["x", "admin-eu"]}), "admin-*", Some(LitellmRole::ProxyAdmin))]
    #[case::falls_to_wildcard(json!({"roles": ["x"]}), "admin-*", Some(LitellmRole::InternalUser))]
    #[case::string_claim_is_no_roles(json!({"roles": "admin-eu"}), "admin-*", None)]
    #[case::empty(json!({"roles": []}), "admin-*", None)]
    fn litellm_role_globs_jwt_roles(
        #[case] token: Value,
        #[case] pattern: &str,
        #[case] expected: Option<LitellmRole>,
    ) {
        assert_eq!(
            litellm_role(&claims(token), &settings(role_map_config(pattern))),
            expected
        );
    }

    #[rstest]
    #[case::list(json!({"user_roles": ["a", "member"]}), true)]
    #[case::string_is_ignored(json!({"user_roles": "member"}), false)]
    #[case::missing(json!({}), false)]
    fn allowed_user_role(#[case] token: Value, #[case] expected: bool) {
        let settings = settings(rbac_config());
        let roles = user_roles(&claims(token), &settings);
        assert_eq!(is_allowed_user_role(roles.as_deref(), &settings), expected);
    }

    #[rstest]
    #[case::unset_upserts(None, true)]
    #[case::valid_upserts(Some(true), true)]
    #[case::invalid_email_blocks(Some(false), false)]
    fn upsert_user_id(#[case] valid_user_email: Option<bool>, #[case] expected: bool) {
        let config = JwtAuthSettings {
            user_id_upsert: true,
            ..JwtAuthSettings::default()
        };
        assert_eq!(
            is_upsert_user_id(&settings(config), valid_user_email),
            expected
        );
    }

    #[rstest]
    #[case::neither(None, None, false)]
    #[case::team_id(Some("t"), None, true)]
    #[case::team_alias(None, Some("a"), true)]
    fn required_team_id(
        #[case] id_field: Option<&str>,
        #[case] alias_field: Option<&str>,
        #[case] expected: bool,
    ) {
        let config = JwtAuthSettings {
            team_id_jwt_field: id_field.map(ClaimPath::from),
            team_alias_jwt_field: alias_field.map(ClaimPath::from),
            ..JwtAuthSettings::default()
        };
        assert_eq!(is_required_team_id(&settings(config)), expected);
    }
}
