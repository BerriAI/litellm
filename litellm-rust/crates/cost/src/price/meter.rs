use std::num::NonZeroU64;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Meter {
    Input(InputUnit),
    Output(OutputUnit),
    CacheRead {
        kind: TokenKind,
        mode: CacheReadMode,
    },
    CacheWrite {
        kind: TokenKind,
        ttl: CacheTtl,
    },
    Feature(Feature),
    WallClockSecond,
    WebSearch(SearchContextSize),
    Guardrail(GuardrailUnit),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TokenKind {
    Text,
    Audio,
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InputUnit {
    Token(TokenKind),
    Character,
    Image,
    Pixel,
    Query { over_results: Option<NonZeroU64> },
    Request,
    Second,
    TextMessage,
    AudioSecond,
    VideoSecond { over_seconds: Option<NonZeroU64> },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OutputUnit {
    Token(TokenKind),
    ReasoningToken,
    Character,
    Image(Option<ImageSize>),
    Pixel,
    Second(Option<Resolution>),
    VideoSecond,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheReadMode {
    Implicit,
    Explicit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheTtl {
    Default,
    OneHour,
    TwelveHours,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Feature {
    AnnotationPage,
    CitationToken,
    CodeInterpreterSession,
    GoogleMapsGroundingQuery,
    OcrCredit,
    OcrPage,
    /// `browser_open_cost_per_request`, Groq `browser.open`
    BrowserOpen,
    /// `cost_per_additional_result`, Parallel AI results past the ones the query price includes
    AdditionalResult,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SearchContextSize {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GuardrailUnit(Box<str>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageSize(Box<str>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Resolution(Box<str>);

impl Meter {
    pub(crate) const TEXT_INPUT: Self = Self::Input(InputUnit::Token(TokenKind::Text));

    pub(crate) fn sets_prompt_band(&self) -> bool {
        matches!(
            self,
            Self::Input(InputUnit::Token(TokenKind::Text))
                | Self::Output(OutputUnit::Token(TokenKind::Text))
                | Self::CacheRead {
                    kind: TokenKind::Text,
                    mode: CacheReadMode::Implicit,
                }
                | Self::CacheWrite {
                    kind: TokenKind::Text,
                    ..
                }
        )
    }

    pub(crate) fn fallback(&self) -> Option<Self> {
        match self {
            Self::CacheRead {
                kind: TokenKind::Text,
                mode: CacheReadMode::Implicit,
            }
            | Self::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            } => Some(Self::TEXT_INPUT),
            Self::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::OneHour | CacheTtl::TwelveHours,
            } => Some(Self::CacheWrite {
                kind: TokenKind::Text,
                ttl: CacheTtl::Default,
            }),
            _ => None,
        }
    }
}

impl GuardrailUnit {
    pub fn new(counter: &str) -> Self {
        Self(counter.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl ImageSize {
    pub fn new(label: &str) -> Option<Self> {
        let number = label.strip_suffix('K').unwrap_or(label);
        let (whole, fraction) = number.split_once('.').unwrap_or((number, "0"));
        (is_digits(whole) && is_digits(fraction)).then(|| Self(label.into()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Resolution {
    pub fn new(label: &str) -> Option<Self> {
        let digits = label.strip_suffix(['p', 'k'])?;
        is_digits(digits).then(|| Self(label.into()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::kilo("1K", true)]
    #[case::half_kilo("0.5K", true)]
    #[case::pixels("1024", true)]
    #[case::word("big", false)]
    #[case::bare_suffix("K", false)]
    #[case::trailing_dot("1.K", false)]
    fn image_size_labels_are_numbers_with_an_optional_k(#[case] label: &str, #[case] valid: bool) {
        assert_eq!(ImageSize::new(label).is_some(), valid);
    }

    #[rstest]
    #[case::progressive("720p", true)]
    #[case::kilo("4k", true)]
    #[case::bare("720", false)]
    #[case::word("hd", false)]
    fn resolution_labels_are_numbers_with_p_or_k(#[case] label: &str, #[case] valid: bool) {
        assert_eq!(Resolution::new(label).is_some(), valid);
    }
}
