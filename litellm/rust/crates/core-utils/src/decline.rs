use std::borrow::Cow;

/// There is no Rust way to do this, found before any effect another implementation would repeat.
///
/// It bubbles up unchanged to the edge that dispatched the work, which decides what it means
/// there: the SDK runs legacy Python, and an edge with nothing else to run answers with an error
/// carrying `code`. It comes from construction, for anything a config
/// decides, or from a route's `plan`, once per request. Nothing that writes may return one.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("declined by the rust path ({code}): {detail}")]
pub struct Decline {
    code: &'static str,
    detail: Cow<'static, str>,
}

impl Decline {
    /// `code` is stable and goes into a response header: lowercase words joined by `.`, named by
    /// layer (`config.guardrails`, `body.field`, `llms.streaming`). `detail` is for the log.
    /// Declare every decline as a `const`, so a malformed code does not compile.
    pub const fn new(code: &'static str, detail: &'static str) -> Self {
        assert!(
            is_code(code),
            "a decline code is words of a-z, 0-9 and _ joined by single dots"
        );
        Self {
            code,
            detail: Cow::Borrowed(detail),
        }
    }

    /// The same decline naming the setting, field or plan it is about.
    #[must_use]
    pub fn with_detail(self, detail: impl Into<Cow<'static, str>>) -> Self {
        Self {
            detail: detail.into(),
            ..self
        }
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }
}

const fn is_code(code: &str) -> bool {
    let bytes = code.as_bytes();
    let mut index = 0;
    let mut word_open = false;
    while index < bytes.len() {
        word_open = match bytes[index] {
            b'a'..=b'z' | b'0'..=b'9' | b'_' => true,
            b'.' if word_open => false,
            _ => return false,
        };
        index += 1;
    }
    word_open
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::one_word("route", true)]
    #[case::dotted("config.guardrails", true)]
    #[case::digits_and_underscores("body.too_large2", true)]
    #[case::empty("", false)]
    #[case::leading_dot(".route", false)]
    #[case::trailing_dot("route.", false)]
    #[case::double_dot("body..field", false)]
    #[case::uppercase("Route", false)]
    #[case::header_breaking("route\r\nx: y", false)]
    #[case::separator("route;x", false)]
    fn a_code_is_lowercase_words_joined_by_dots(#[case] code: &str, #[case] valid: bool) {
        assert_eq!(is_code(code), valid);
    }

    #[rstest]
    fn a_detail_replaces_the_text_and_keeps_the_code() {
        const FIELD: Decline = Decline::new("body.field", "a field the route does not read");
        let named = FIELD.with_detail("field `mock_response`");
        assert_eq!(named.code(), "body.field");
        assert_eq!(named.detail(), "field `mock_response`");
        assert_eq!(
            named.to_string(),
            "declined by the rust path (body.field): field `mock_response`"
        );
    }
}
