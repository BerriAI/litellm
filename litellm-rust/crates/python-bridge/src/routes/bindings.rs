use litellm_python_interop::InvocationMode;
use strum::{Display, EnumIter, IntoStaticStr};

/// A method name paired inseparably with the mode used to drive it.
///
/// Keeping the two together is what prevents a `("prepare", Await)`-style
/// mismatch. Callers never assemble these by hand; they ask a catalog entry
/// ([`BoundaryMethod`] or [`LoggingMethod`]) to `resolve` one.
pub(crate) struct MethodBinding {
    pub(crate) name: &'static str,
    pub(crate) mode: InvocationMode,
}

/// OCR route boundary methods: the Python operations invoked with
/// [`litellm_python_interop::PreparedCall`] to carry a request through
/// preparation, encoding, transport and finalization.
///
/// The `asynchronous` flag drives both the method *name* and its invocation
/// mode together, so a sync/async pair cannot desynchronize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, Display, IntoStaticStr)]
pub(crate) enum BoundaryMethod {
    Prepare,
    Encode,
    Finish,
}

impl BoundaryMethod {
    pub(crate) fn resolve(self, asynchronous: bool) -> MethodBinding {
        match (self, asynchronous) {
            (Self::Prepare, true) => MethodBinding {
                name: "aprepare",
                mode: InvocationMode::Await,
            },
            (Self::Prepare, false) => MethodBinding {
                name: "prepare",
                mode: InvocationMode::Direct,
            },
            (Self::Encode, _) => MethodBinding {
                name: "encode",
                mode: InvocationMode::Direct,
            },
            (Self::Finish, true) => MethodBinding {
                name: "afinish",
                mode: InvocationMode::Await,
            },
            (Self::Finish, false) => MethodBinding {
                name: "finish",
                mode: InvocationMode::Direct,
            },
        }
    }
}

/// Bound methods on the Python `Logging` object that the bridge invokes.
///
/// Sync hooks (`pre_call`, `success_handler`, `failure_handler`) run inline
/// even on async routes; async hooks return a coroutine for the caller's loop
/// to drive. This mirrors the split LiteLLM keeps in Python between
/// `dynamic_success_callbacks` / `dynamic_async_success_callbacks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, Display, IntoStaticStr)]
pub(crate) enum LoggingMethod {
    PreCall,
    PostCall,
    SuccessHandler,
    FailureHandler,
    AsyncSuccessHandler,
    AsyncFailureHandler,
}

// `LoggingMethod` is the forward-looking catalog for the retained-callback
// foundation. Nothing in production consumes it yet (the hooks them are
// exercised by the `component_contract` fixtures via a direct `PreparedCall`),
// so `resolve` is only reached from this module's tests for now.
#[cfg_attr(not(test), allow(dead_code))]
impl LoggingMethod {
    pub(crate) fn resolve(self) -> MethodBinding {
        match self {
            Self::PreCall => MethodBinding {
                name: "pre_call",
                mode: InvocationMode::Direct,
            },
            Self::PostCall => MethodBinding {
                name: "post_call",
                mode: InvocationMode::Direct,
            },
            Self::SuccessHandler => MethodBinding {
                name: "success_handler",
                mode: InvocationMode::Direct,
            },
            Self::FailureHandler => MethodBinding {
                name: "failure_handler",
                mode: InvocationMode::Direct,
            },
            Self::AsyncSuccessHandler => MethodBinding {
                name: "async_success_handler",
                mode: InvocationMode::Await,
            },
            Self::AsyncFailureHandler => MethodBinding {
                name: "async_failure_handler",
                mode: InvocationMode::Await,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn boundary_methods_pair_name_and_mode_consistently() {
        for method in BoundaryMethod::iter() {
            let sync = method.resolve(false);
            let asynchronous = method.resolve(true);

            assert!(!sync.name.is_empty());
            assert!(!asynchronous.name.is_empty());

            if method == BoundaryMethod::Encode {
                // `encode` has no async variant; it is always a direct call.
                assert_eq!(sync.name, asynchronous.name);
                assert_eq!(sync.mode, InvocationMode::Direct);
                assert_eq!(asynchronous.mode, InvocationMode::Direct);
            } else {
                assert_eq!(sync.mode, InvocationMode::Direct);
                assert_eq!(asynchronous.mode, InvocationMode::Await);
                let async_name = asynchronous.name.strip_prefix('a').unwrap();
                assert_eq!(sync.name, async_name);
            }
        }
    }

    #[test]
    fn logging_methods_are_direct_unless_async() {
        let mut names = Vec::new();
        for method in LoggingMethod::iter() {
            let binding = method.resolve();
            assert!(!binding.name.is_empty());
            assert!(!names.contains(&binding.name), "duplicate method name");
            names.push(binding.name);

            let is_async = matches!(
                method,
                LoggingMethod::AsyncSuccessHandler | LoggingMethod::AsyncFailureHandler
            );
            assert_eq!(
                binding.mode == InvocationMode::Await,
                is_async,
                "{} must be {}",
                binding.name,
                if is_async { "Await" } else { "Direct" }
            );
        }
    }
}
