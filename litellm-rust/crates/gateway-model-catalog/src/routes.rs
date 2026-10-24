use axum::{
    Json, Router,
    extract::{FromRef, State},
    routing::get,
};
use litellm_gateway_auth::{AccessRequest, AuthenticatedRequest, Error as AuthError};
use serde::Serialize;
use time::format_description::well_known::Rfc3339;

use crate::{CatalogReader, Source};

const PATH: &str = "/model/cost_map/source";

#[derive(Serialize)]
struct SourceResponse {
    source: Source,
    url: Option<String>,
    is_env_forced: bool,
    fallback_reason: Option<String>,
    loaded_at: String,
    source_revision: String,
    etag: Option<String>,
    model_count: usize,
}

pub fn router<S>() -> Router<S>
where
    CatalogReader: FromRef<S>,
    S: Clone + Send + Sync + 'static,
{
    Router::new().route(PATH, get(source))
}

async fn source(
    State(reader): State<CatalogReader>,
    identity: AuthenticatedRequest,
) -> Result<Json<SourceResponse>, AuthError> {
    let request = AccessRequest::Route {
        method: "GET".to_owned(),
        path: PATH.to_owned(),
    };
    let grant = identity.authorize(request.clone()).await?;
    grant.consume(identity.caller(), &request)?;
    let served = reader.current();
    let provenance = &served.provenance;
    Ok(Json(SourceResponse {
        source: provenance.source.clone(),
        url: provenance.url.clone(),
        is_env_forced: provenance.is_env_forced,
        fallback_reason: provenance.fallback_reason.clone(),
        loaded_at: provenance
            .loaded_at
            .format(&Rfc3339)
            .expect("UTC timestamps are RFC 3339 representable"),
        source_revision: served.installed.revision.clone(),
        etag: served.installed.etag.clone(),
        model_count: served.installed.catalog.served().count(),
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_util::future::BoxFuture;
    use litellm_auth_types::SecretValue;
    use litellm_gateway_auth::{
        Auth, AuthFuture, IdentityResolver, MasterKeyAuthenticator, NoAdditionalPolicy,
        Permissions, ResolvedIdentity, SystemClock, VerifiedIdentity,
    };
    use litellm_model_catalog::Catalog;
    use litellm_secrets::source::SecretSource;
    use rstest::rstest;

    use super::*;

    struct FixedPermissions(Permissions);

    impl IdentityResolver for FixedPermissions {
        fn resolve<'a>(
            &'a self,
            identity: &'a VerifiedIdentity,
        ) -> AuthFuture<'a, ResolvedIdentity> {
            Box::pin(async move {
                Ok(ResolvedIdentity {
                    principal: identity.principal.clone(),
                    permissions: self.0.clone(),
                })
            })
        }
    }

    struct NoSecrets;

    impl SecretSource for NoSecrets {
        fn get_secret_str<'a>(
            &'a self,
            _: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
            Box::pin(async { Ok(None) })
        }
    }

    async fn identity(permissions: Permissions) -> AuthenticatedRequest {
        Auth::new(
            Arc::new(MasterKeyAuthenticator::new(
                Some(SecretValue::new("catalog-test-key")),
                Arc::new(NoSecrets),
            )),
            Arc::new(FixedPermissions(permissions)),
            Arc::new(NoAdditionalPolicy),
            Arc::new(SystemClock),
        )
        .authenticate(&SecretValue::new("catalog-test-key"))
        .await
        .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn handler_rejects_a_caller_without_route_permission() {
        let (_, reader) = crate::bundled(
            &crate::CostMapSettings::default(),
            time::OffsetDateTime::now_utc(),
        );
        let request = identity(Permissions::None).await;
        let response = source(State(reader), request).await;
        assert!(matches!(response, Err(AuthError::Forbidden)));
    }

    #[rstest]
    #[tokio::test]
    async fn response_matches_shared_python_contract_fixture() {
        let (writer, reader) = crate::bundled(
            &crate::CostMapSettings::default(),
            time::OffsetDateTime::from_unix_timestamp(1_767_323_045).unwrap(),
        );
        let mut served = (*writer.current()).clone();
        let catalog = Catalog::parse(
            br#"{"fixture-model":{"litellm_provider":"openai","input_cost_per_token":0.000001}}"#,
        )
        .unwrap();
        assert!(catalog.errors.is_empty());
        served.installed.catalog = Arc::new(catalog.value);
        served.installed.revision = "fixture-revision".to_owned();
        writer.install(served);
        let request = identity(Permissions::All).await;
        let response = source(State(reader), request).await.unwrap();
        let response = serde_json::to_value(response.0).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cost_map_source.json")).unwrap();
        assert_eq!(response, fixture);
    }
}
