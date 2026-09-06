use crate::request_context::RequestCapabilities;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeRouteDecline {
    UnsupportedProvider,
    Streaming,
    AgenticHook,
    CustomClient,
}

impl NativeRouteDecline {
    pub const fn reason(self) -> &'static str {
        match self {
            Self::UnsupportedProvider => "unsupported native provider",
            Self::Streaming => "native streaming is unavailable",
            Self::AgenticHook => "native agentic hooks are unavailable",
            Self::CustomClient => "native custom clients are unavailable",
        }
    }
}

pub fn native_route_decline(
    provider_supported: bool,
    capabilities: &RequestCapabilities,
) -> Option<NativeRouteDecline> {
    if !provider_supported {
        return Some(NativeRouteDecline::UnsupportedProvider);
    }
    if capabilities.stream {
        return Some(NativeRouteDecline::Streaming);
    }
    if capabilities.has_agentic_hook {
        return Some(NativeRouteDecline::AgenticHook);
    }
    if capabilities.has_custom_client {
        return Some(NativeRouteDecline::CustomClient);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_preserves_precedence_and_accepts_supported_unary_calls() {
        let all_unsupported = RequestCapabilities {
            stream: true,
            has_agentic_hook: true,
            has_custom_client: true,
            request_format: Some("native".into()),
            ..Default::default()
        };
        assert_eq!(
            native_route_decline(false, &all_unsupported),
            Some(NativeRouteDecline::UnsupportedProvider)
        );
        assert_eq!(
            native_route_decline(true, &all_unsupported),
            Some(NativeRouteDecline::Streaming)
        );
        assert_eq!(
            native_route_decline(true, &RequestCapabilities::default()),
            None
        );
    }

    #[test]
    fn admission_rejects_each_unsupported_capability() {
        let cases = [
            (
                RequestCapabilities {
                    has_agentic_hook: true,
                    ..Default::default()
                },
                NativeRouteDecline::AgenticHook,
            ),
            (
                RequestCapabilities {
                    has_custom_client: true,
                    ..Default::default()
                },
                NativeRouteDecline::CustomClient,
            ),
        ];
        for (capabilities, expected) in cases {
            assert_eq!(native_route_decline(true, &capabilities), Some(expected));
        }
    }
}
