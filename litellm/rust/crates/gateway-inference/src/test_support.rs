//! Callers for the inline tests, over rows that hold a key, which the native host's do not yet,
//! and the deployments they call.

use std::sync::Arc;

use litellm_gateway_auth_core::{
    Affiliation, Caller, Credential, KeyHash, Rows, authn::key::KeyRecord,
};
use litellm_router_core::{Deployment, DeploymentSpec, RouterConfig};
use litellm_router_types::RouterSettings;
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Held;

impl Rows for Held {
    type Key = KeyRecord;
    type User = ();
    type Team = ();
    type EndUser = ();
    type ObjectPermission = ();
    type Project = ();
    type Membership = ();
    type Org = ();
    type Agent = ();
}

/// A key whose row is stored under `stored` and says `is_proxy_admin`.
pub(crate) fn key(stored: &str, is_proxy_admin: bool) -> Caller<Held> {
    Caller {
        credential: Credential::Key {
            key: KeyRecord {
                token: Some(KeyHash::new(stored)),
                is_proxy_admin,
                ..KeyRecord::default()
            },
        },
        affiliation: Affiliation::nobody(),
        end_user: (),
    }
}

/// The deployment the router registers for `litellm_params` and `model_info`.
pub(crate) fn deployment(litellm_params: Value, model_info: Value) -> Arc<Deployment> {
    let (config, dynamic) = RouterConfig::from_settings(&RouterSettings::default()).unwrap();
    let router = litellm_router_core::Router::new(
        RouterConfig {
            model_list: vec![DeploymentSpec {
                model_name: "public-model".into(),
                litellm_params: litellm_params.as_object().cloned().unwrap(),
                model_info: model_info.as_object().cloned().unwrap(),
                ..DeploymentSpec::default()
            }],
            ..config
        },
        dynamic,
    )
    .unwrap();
    router.registry().group("public-model").unwrap().remove(0)
}
