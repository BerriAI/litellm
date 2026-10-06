use std::sync::Arc;

use futures_util::future::BoxFuture;
use litellm_router::{Deployment, Router, RouterHooks};
use rstest::{fixture, rstest};

#[fixture]
fn router() -> Router {
    Router::from_iter([(
        "public-model".into(),
        Deployment {
            model: "provider/model".into(),
            ..Default::default()
        },
    )])
}

struct Reject;

impl RouterHooks for Reject {
    fn filter_deployments<'h, 'd: 'h>(
        &'h self,
        model: &'h str,
        deployments: Vec<&'d Deployment>,
    ) -> BoxFuture<'h, Vec<&'d Deployment>> {
        assert_eq!(model, "public-model");
        assert_eq!(deployments.len(), 1);
        assert_eq!(deployments[0].model, "provider/model");
        Box::pin(async { Vec::new() })
    }
}

#[rstest]
#[case::known("public-model", Some("provider/model"))]
#[case::unknown("missing", None)]
#[tokio::test]
async fn default_hooks_select_by_model_name(
    router: Router,
    #[case] model: &str,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        router
            .select(model)
            .await
            .map(|deployment| deployment.model.as_str()),
        expected,
    );
}

#[rstest]
#[tokio::test]
async fn selection_uses_filtered_candidates(router: Router) {
    assert!(
        router
            .with_hooks(Arc::new(Reject))
            .select("public-model")
            .await
            .is_none()
    );
}
