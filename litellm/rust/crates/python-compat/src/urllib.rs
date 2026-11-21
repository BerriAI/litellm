//! `urllib.parse.quote` over the UTF-8 bytes of a text, and `urllib.parse.urlsplit`.

/// `urllib.parse.quote(text, safe=safe)`: ASCII letters, digits and `_.-~` are never quoted,
/// the ASCII characters in `safe` are kept, and every other byte becomes `%XX`.
pub fn quote(text: &str, safe: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric()
            || matches!(byte, b'_' | b'.' | b'-' | b'~')
            || (byte.is_ascii() && safe.as_bytes().contains(&byte))
        {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

/// `urllib.parse.urlsplit(url)` over text: the scheme, the network location, the path, the
/// query and the fragment, each borrowed from `url` as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitUrl<'a> {
    /// `scheme:` and `//netloc`, as written.
    pub origin: &'a str,
    pub scheme: &'a str,
    pub netloc: &'a str,
    pub path: &'a str,
    pub query: &'a str,
    pub fragment: &'a str,
}

fn is_scheme(text: &str) -> bool {
    text.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
        && text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
}

/// `urllib.parse.urlsplit(url)`.
pub fn urlsplit(url: &str) -> SplitUrl<'_> {
    let (before_fragment, fragment) = url.split_once('#').unwrap_or((url, ""));
    let (before_query, query) = before_fragment
        .split_once('?')
        .unwrap_or((before_fragment, ""));
    let (scheme, after_scheme) = match before_query.split_once(':') {
        Some((scheme, rest)) if is_scheme(scheme) => (scheme, rest),
        _ => ("", before_query),
    };
    let (netloc, path) = match after_scheme.strip_prefix("//") {
        Some(rest) => rest.split_at(rest.find('/').unwrap_or(rest.len())),
        None => ("", after_scheme),
    };
    let origin = &before_query[..before_query.len() - path.len()];
    SplitUrl {
        origin,
        scheme,
        netloc,
        path,
        query,
        fragment,
    }
}

impl SplitUrl<'_> {
    /// `urlunsplit` with `path` and `query` in place of the URL's own: its scheme lowercased
    /// and an empty query or fragment dropped.
    pub fn with(&self, path: &str, query: &str) -> String {
        let mut url = format!(
            "{}{}{path}",
            self.scheme.to_ascii_lowercase(),
            &self.origin[self.scheme.len()..]
        );
        if !query.is_empty() {
            url = format!("{url}?{query}");
        }
        if !self.fragment.is_empty() {
            url = format!("{url}#{}", self.fragment);
        }
        url
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::slash_kept("a/b c", "/", "a/b%20c")]
    #[case::slash_quoted("a/b", "", "a%2Fb")]
    #[case::unreserved_kept("a_b.c-d~e", "", "a_b.c-d~e")]
    #[case::utf8_bytes("é", "/", "%C3%A9")]
    fn quote_keeps_unreserved_and_safe_characters(
        #[case] text: &str,
        #[case] safe: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(quote(text, safe), expected);
    }

    #[rstest]
    #[case::full(
        "HTTPS://h:1/p/q?x=1#f",
        SplitUrl { origin: "HTTPS://h:1", scheme: "HTTPS", netloc: "h:1", path: "/p/q", query: "x=1", fragment: "f" }
    )]
    #[case::no_scheme("h/p", SplitUrl { origin: "", scheme: "", netloc: "", path: "h/p", query: "", fragment: "" })]
    #[case::scheme_only("http:", SplitUrl { origin: "http:", scheme: "http", netloc: "", path: "", query: "", fragment: "" })]
    #[case::not_a_scheme("1http://h/p", SplitUrl { origin: "", scheme: "", netloc: "", path: "1http://h/p", query: "", fragment: "" })]
    fn urlsplit_splits_as_python_does(#[case] url: &str, #[case] expected: SplitUrl<'_>) {
        assert_eq!(urlsplit(url), expected);
    }

    #[rstest]
    #[case::new_query("HTTPS://h/p#f", "/q", "a=1", "https://h/q?a=1#f")]
    #[case::no_query("http://h/p?x", "/p", "", "http://h/p")]
    fn with_rebuilds_the_url(
        #[case] url: &str,
        #[case] path: &str,
        #[case] query: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(urlsplit(url).with(path, query), expected);
    }
}
