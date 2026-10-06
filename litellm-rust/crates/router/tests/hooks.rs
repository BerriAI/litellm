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
    async fn filter_deployments<'a>(
        &self,
        model: &str,
        deployments: Vec<&'a Deployment>,
    ) -> Vec<&'a Deployment> {
        assert_eq!(model, "public-model");
        assert_eq!(deployments.len(), 1);
        assert_eq!(deployments[0].model, "provider/model");
        Vec::new()
    }
}

#[rstest]
#[case::known("public-model")]
#[case::unknown("missing")]
#[tokio::test]
async fn default_hooks_preserve_model_lookup(router: Router, #[case] model: &str) {
    assert_eq!(
        router
            .select(model)
            .await
            .map(|deployment| &deployment.model),
        router.get(model).map(|deployment| &deployment.model),
    );
}

#[rstest]
#[tokio::test]
async fn selection_uses_filtered_candidates(router: Router) {
    assert!(
        router
            .with_hooks(Reject)
            .select("public-model")
            .await
            .is_none()
    );
}
