//! The proxy's call type for the route matched at a path, for whoever has to tell it before the
//! route runs, as the gateway's `admit` does to know which bodies only Python estimates.

use crate::CallType;

/// The call type the proxy gives the route matched at `path`, which decides what a token estimate
/// reads from the body. A path no inference route serves has none.
pub fn call_type(path: &str) -> Option<CallType> {
    if path.ends_with("/messages") {
        Some(CallType::AnthropicMessages)
    } else if path.ends_with("/ocr") {
        Some(CallType::Aocr)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::messages("/v1/messages", "anthropic_messages")]
    #[case::ocr("/v1/ocr", "aocr")]
    #[case::any_other("/v1/embeddings", "")]
    fn each_route_is_estimated_as_the_proxys_call_type(#[case] path: &str, #[case] expected: &str) {
        assert_eq!(call_type(path).map_or("", <&str>::from), expected);
    }
}
