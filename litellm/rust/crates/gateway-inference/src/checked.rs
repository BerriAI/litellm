//! A request to an inference route whose caller is authenticated and whose body the proxy's
//! request body check accepted, Python's `is_request_body_safe` from `pre_db_read_auth_checks`.
//!
//! [`Checked`] is one extractor for the caller and the body, so a route that takes it cannot read
//! a body that was not checked. The check is `gateway-auth-core`'s `body::check`, which decides
//! on data: this file reads the body into what the check reads, resolves the model groups it can
//! reach through the router, and reads the stored credentials it names through the host.

use std::sync::Arc;

use axum::extract::{FromRequest, FromRequestParts, Request};
use litellm_config::GeneralSettings;
use litellm_gateway_auth_core::{
    Caller, StoredKey,
    body::{
        self, Deferred, Deployment, HostTables, Names, OptIn, Params, Policy, Reachable, Surface,
    },
};
use serde_json::{Map, Value};

use crate::{Error, Gateway, JsonObject, request::InferenceBody, routing::Host, serve::Call};

const CLIENTSIDE_AUTH_PARAMS: &str = "configurable_clientside_auth_params";

/// What the body check reads besides the request: the parameter names it judges and what the
/// admin allowed callers to set.
#[derive(Clone, Debug)]
pub struct BodyCheck {
    names: Names,
    policy: Policy,
}

impl BodyCheck {
    /// `general_settings.allow_client_side_credentials`. The S3 allowlists and
    /// `provider_url_destination_allowed_hosts` are left empty, and so are the name tables
    /// Python still owns (callback, pricing, federation and upstream-bound parameters), so only
    /// the check's own override names are judged until they are ported.
    pub fn new(general: &GeneralSettings) -> Self {
        Self {
            names: Names::new(HostTables::default()),
            policy: Policy {
                allow_client_side_credentials: general.allow_client_side_credentials,
                ..Policy::default()
            },
        }
    }
}

/// A body the check reads its fields from.
pub(crate) trait CheckedBody {
    fn fields(&self) -> &Map<String, Value>;
}

impl CheckedBody for JsonObject {
    fn fields(&self) -> &Map<String, Value> {
        &self.0
    }
}

impl CheckedBody for InferenceBody {
    fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }
}

/// The caller and the body of one request. A body that could not be read, or that the check
/// refused, is the route's error, so the route answers it in its own format.
pub struct Checked<H: Host, B> {
    pub call: Call<H>,
    pub body: Result<B, Error>,
}

impl<H, B> FromRequest<Arc<Gateway<H>>> for Checked<H, B>
where
    H: Host,
    B: FromRequest<Arc<Gateway<H>>, Rejection = Error> + CheckedBody + Send,
{
    type Rejection = litellm_gateway_auth::Error;

    async fn from_request(
        request: Request,
        gateway: &Arc<Gateway<H>>,
    ) -> Result<Self, Self::Rejection> {
        let (mut parts, body) = request.into_parts();
        let call = Call::from_request_parts(&mut parts, gateway).await?;
        let route = parts.uri.path().to_owned();
        let body = match B::from_request(Request::from_parts(parts, body), gateway).await {
            Ok(body) => check(&call, &route, body.fields()).await.map(|()| body),
            Err(error) => Err(error),
        };
        Ok(Self { call, body })
    }
}

/// `is_request_body_safe` for the body of a call to `route`. Model checks and the S3 bucket check
/// are never deferred, since neither route checks a team's buckets in its handler.
async fn check<H: Host>(
    call: &Call<H>,
    route: &str,
    fields: &Map<String, Value>,
) -> Result<(), Error> {
    let gateway = &call.gateway;
    let BodyCheck { names, policy } = &gateway.body_check;
    let surface = Surface::of(Some(route));
    let params = Params::read(fields, names, &surface);
    let request = body::Request {
        body: &params,
        surface,
        model: fields
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        route_model: None,
        deferred: Deferred::default(),
    };
    let needs = body::needs(&request, names, policy);
    let team = match &*call.identity {
        Caller::MasterKey => None,
        Caller::Key(context) => context.key.record().team_id.as_deref(),
    };
    let reachable = Reachable {
        groups: gateway
            .router
            .opt_in_groups(&needs.models, team)
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|deployment| opt_in_deployment(deployment))
                    .collect()
            })
            .collect(),
    };
    let credentials = gateway
        .host
        .credentials(&needs.credentials)
        .await
        .map_err(Error::Credentials)?;
    body::check(&request, &reachable, &credentials, names, policy).map_err(Error::BodyRejected)
}

/// A deployment as the check reads it: its group, its provider and what it lets a caller set.
fn opt_in_deployment(deployment: &litellm_router_core::registry::Deployment) -> Deployment {
    Deployment {
        model_name: deployment.model_name.clone(),
        provider: Some(deployment.provider().to_owned()),
        opt_in: opt_in(deployment.raw_params.get(CLIENTSIDE_AUTH_PARAMS)),
    }
}

/// `configurable_clientside_auth_params`: parameter names, and `{"api_base": pattern}` entries.
/// An entry of any other shape grants nothing.
fn opt_in(written: Option<&Value>) -> Option<OptIn> {
    let entries = match written? {
        Value::Null => return None,
        Value::Array(entries) => entries.as_slice(),
        _ => &[],
    };
    Some(OptIn {
        params: entries
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        api_base_patterns: entries
            .iter()
            .filter_map(|entry| entry.get("api_base")?.as_str())
            .map(str::to_owned)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::unset(None, None)]
    #[case::null(Some(json!(null)), None)]
    #[case::names_and_patterns(
        Some(json!(["client_id", {"api_base": "https://own\\.example/.*"}, 3])),
        Some(OptIn { params: vec!["client_id".to_owned()], api_base_patterns: vec!["https://own\\.example/.*".to_owned()] })
    )]
    #[case::not_a_list(Some(json!("client_id")), Some(OptIn::default()))]
    fn a_deployments_opt_in_is_read_as_python_reads_it(
        #[case] written: Option<Value>,
        #[case] expected: Option<OptIn>,
    ) {
        assert_eq!(opt_in(written.as_ref()), expected);
    }
}
