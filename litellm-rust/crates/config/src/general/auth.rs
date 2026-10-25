use litellm_auth_types::SecretValue;

use crate::{Object, Raw, Spelled};

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AuthSettings {
    pub litellm_key_header_name: Option<String>,
    pub enable_jwt_auth: Option<Spelled<bool>>,
    pub litellm_jwtauth: Option<Object>,
    pub enable_oauth2_auth: Option<Spelled<bool>>,
    pub enable_oauth2_proxy_auth: Option<Spelled<bool>>,
    pub oauth2_config_mappings: Option<Object>,
    pub custom_auth: Option<String>,
    pub custom_auth_run_common_checks: Option<Spelled<bool>>,
    pub allow_user_auth: Option<Spelled<bool>>,
    pub allowed_routes: Option<Box<[String]>>,
    pub admin_only_routes: Option<Box<[String]>>,
    pub public_routes: Option<Box<[String]>>,
    pub allowed_ips: Option<Box<[String]>>,
    pub use_x_forwarded_for: Option<Spelled<bool>>,
    pub trusted_proxy_ranges: Option<Box<[String]>>,
    pub user_header_name: Option<String>,
    pub user_header_mappings: Option<Box<[UserHeaderMapping]>>,
    pub role_permissions: Option<Box<[RoleBasedPermissions]>>,
    pub enforce_rbac: Option<Spelled<bool>>,
    pub allow_client_side_credentials: Option<Spelled<bool>>,
    pub enforce_user_param: Option<Spelled<bool>>,
    pub reject_clientside_metadata_tags: Option<Spelled<bool>>,
    pub enforced_params: Option<Box<[String]>>,
    pub service_account_settings: Option<ServiceAccountSettings>,
    pub litellm_license: Option<SecretValue>,
    pub control_plane_url: Option<String>,
    pub key_management_system: Option<String>,
    pub key_management_settings: Option<KeyManagementSettings>,
    pub use_google_kms: Option<Spelled<bool>>,
    pub use_azure_key_vault: Option<Spelled<bool>>,
    pub encryption_algorithm: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct UserHeaderMapping {
    pub header_name: Option<String>,
    pub litellm_user_role: Option<String>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct RoleBasedPermissions {
    pub role: Option<String>,
    pub models: Option<Box<[String]>>,
    pub routes: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ServiceAccountSettings {
    pub enforced_params: Option<Box<[String]>>,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct KeyManagementSettings {
    pub hosted_keys: Option<Box<[Raw]>>,
    pub store_virtual_keys: Option<Spelled<bool>>,
    pub prefix_for_stored_virtual_keys: Option<String>,
    pub access_mode: Option<String>,
    pub primary_secret_name: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Object>,
    pub kms_key_id: Option<String>,
    pub custom_secret_manager: Option<String>,
    pub aws_region_name: Option<String>,
    pub aws_role_name: Option<String>,
    pub aws_session_name: Option<String>,
    pub aws_external_id: Option<SecretValue>,
    pub aws_profile_name: Option<String>,
    pub aws_web_identity_token: Option<SecretValue>,
    pub aws_sts_endpoint: Option<String>,
    pub replica_regions: Option<Box<[String]>>,
}
