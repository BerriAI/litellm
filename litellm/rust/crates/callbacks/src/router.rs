use std::{error::Error, fmt, sync::Arc};

use futures_util::future::BoxFuture;
use num_bigint::BigInt;

use crate::record::RouterLimits;

#[derive(Clone, Debug, PartialEq)]
pub enum UsageAmount {
    Int(BigInt),
    Float(f64),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RemainingUsage {
    pub headers: Vec<(String, UsageAmount)>,
}

impl RemainingUsage {
    pub fn get(&self, name: &str) -> Option<&UsageAmount> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, amount)| amount)
    }
}

pub type RouterReadError = Box<dyn Error + Send + Sync>;

/// Reads a group's remaining usage from the limits Python resolved for it: the counter keys, the
/// store reads and the arithmetic, with no call back into Python.
pub trait RouterUsage: Send + Sync + 'static {
    fn remaining(
        &self,
        model_group: &str,
        limits: &RouterLimits,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>>;
}

#[derive(Clone)]
pub struct RouterRef(Arc<dyn RouterUsage>);

impl RouterRef {
    pub fn new(router: Arc<dyn RouterUsage>) -> Self {
        Self(router)
    }

    pub fn remaining(
        &self,
        model_group: &str,
        limits: &RouterLimits,
    ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
        self.0.remaining(model_group, limits)
    }
}

impl fmt::Debug for RouterRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RouterRef").finish_non_exhaustive()
    }
}

impl PartialEq for RouterRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use rstest::rstest;

    use super::*;

    struct Fixed;

    impl RouterUsage for Fixed {
        fn remaining(
            &self,
            model_group: &str,
            limits: &RouterLimits,
        ) -> BoxFuture<'static, Result<RemainingUsage, RouterReadError>> {
            let headers = vec![(
                format!("x-{model_group}"),
                UsageAmount::Int(BigInt::from(limits.reset.unwrap_or_default())),
            )];
            Box::pin(async move { Ok(RemainingUsage { headers }) })
        }
    }

    #[rstest]
    fn a_router_reference_reads_through_the_trait_object() {
        let router = RouterRef::new(Arc::new(Fixed));
        let remaining = router
            .remaining(
                "g",
                &RouterLimits {
                    reset: Some(7),
                    ..RouterLimits::default()
                },
            )
            .now_or_never()
            .unwrap_or_else(|| panic!("the fixed router answers at once"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            remaining.get("x-g"),
            Some(&UsageAmount::Int(BigInt::from(7)))
        );
        assert_eq!(remaining.get("x-h"), None);
        assert_eq!(router, router.clone());
        assert_ne!(router, RouterRef::new(Arc::new(Fixed)));
    }
}
