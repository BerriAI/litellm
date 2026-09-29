#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndpointPath(&'static str);

impl EndpointPath {
    pub const fn new(path: &'static str) -> Self {
        let bytes = path.as_bytes();
        assert!(
            bytes.len() > 1 && bytes[0] == b'/',
            "endpoint path must start with / and contain segments"
        );
        let mut start = 1;
        let mut index = 1;
        while index <= bytes.len() {
            if index == bytes.len() || bytes[index] == b'/' {
                assert!(index > start, "endpoint path contains an empty segment");
                assert!(
                    !(index - start == 1 && bytes[start] == b'.'),
                    "endpoint path contains a dot segment"
                );
                assert!(
                    !(index - start == 2 && bytes[start] == b'.' && bytes[start + 1] == b'.'),
                    "endpoint path contains a dot segment"
                );
                start = index + 1;
            } else {
                let byte = bytes[index];
                assert!(
                    byte.is_ascii_alphanumeric()
                        || matches!(
                            byte,
                            b'-' | b'.'
                                | b'_'
                                | b'~'
                                | b'!'
                                | b'$'
                                | b'&'
                                | b'\''
                                | b'('
                                | b')'
                                | b'*'
                                | b'+'
                                | b','
                                | b';'
                                | b'='
                                | b':'
                                | b'@'
                        ),
                    "endpoint path contains invalid characters"
                );
            }
            index += 1;
        }
        Self(path)
    }

    pub fn segments<'a>(self) -> impl Iterator<Item = &'a str> {
        let path: &'a str = self.0;
        path[1..].split('/')
    }

    pub fn is_suffix_of(self, path: &str) -> bool {
        path.trim_end_matches('/').ends_with(self.0)
    }

    pub fn matches(self, path: &str) -> bool {
        self.segments()
            .eq(path.split('/').filter(|segment| !segment.is_empty()))
    }
}

/// ```
/// litellm_llms_types::wire_endpoint_enum! {
///     pub enum ExampleEndpoint { Operation => "/v1/operation" }
/// }
/// assert!(ExampleEndpoint::Operation.path().matches("/v1/operation"));
/// ```
/// ```compile_fail
/// litellm_llms_types::wire_endpoint_enum! {
///     pub enum InvalidEndpoint { Operation => "/v1/../operation" }
/// }
/// ```
#[macro_export]
macro_rules! wire_endpoint_enum {
    ($vis:vis enum $name:ident { $($variant:ident => $path:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
        $vis enum $name {
            $(#[serde(rename = $path)] $variant),+
        }
        impl $name {
            pub const fn path(self) -> $crate::endpoint::EndpointPath {
                match self {
                    $(Self::$variant => {
                        const PATH: $crate::endpoint::EndpointPath = $crate::endpoint::EndpointPath::new($path);
                        PATH
                    }),+
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::relative("v1/ocr")]
    #[case::authority("//host/path")]
    #[case::query("/v1?x=1")]
    #[case::fragment("/v1#part")]
    #[case::escape("/v1/%2f")]
    #[case::empty("/v1//ocr")]
    #[case::dot("/v1/..")]
    #[case::backslash("/v1\\ocr")]
    fn rejects_invalid_paths(#[case] path: &'static str) {
        assert!(std::panic::catch_unwind(|| EndpointPath::new(path)).is_err());
    }
}
