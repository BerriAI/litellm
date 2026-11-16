//! `urllib.parse.quote(text, safe=safe)` over the UTF-8 bytes of `text`, and `urlsplit`, `urlparse`
//! and `urlunparse` over a URL string.

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

const UNSAFE_URL_CHARS: [char; 3] = ['\t', '\r', '\n'];
const USES_PARAMS: [&str; 16] = [
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu",
    "sip", "sips", "mms", "sftp", "tel",
];
const USES_NETLOC: [&str; 27] = [
    "",
    "ftp",
    "http",
    "gopher",
    "nntp",
    "telnet",
    "imap",
    "wais",
    "file",
    "mms",
    "https",
    "shttp",
    "snews",
    "prospero",
    "rtsp",
    "rtsps",
    "rtspu",
    "rsync",
    "svn",
    "svn+ssh",
    "sftp",
    "nfs",
    "git",
    "git+ssh",
    "ws",
    "wss",
    "itms-services",
];

/// `urllib.parse.ParseResult`. `params` is always empty for [`urlsplit`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedUrl {
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub params: String,
    pub query: String,
    pub fragment: String,
}

impl ParsedUrl {
    /// `urlunparse(self)`, which is `geturl()` on a `ParseResult`.
    pub fn geturl(&self) -> String {
        let path = match self.params.is_empty() {
            true => self.path.clone(),
            false => format!("{};{}", self.path, self.params),
        };
        let path = if !self.netloc.is_empty() {
            let rooted = match path.is_empty() || path.starts_with('/') {
                true => path,
                false => format!("/{path}"),
            };
            format!("//{}{rooted}", self.netloc)
        } else if path.starts_with("//")
            || (!self.scheme.is_empty()
                && USES_NETLOC.contains(&self.scheme.as_str())
                && (path.is_empty() || path.starts_with('/')))
        {
            format!("//{path}")
        } else {
            path
        };
        let scheme = match self.scheme.is_empty() {
            true => String::new(),
            false => format!("{}:", self.scheme),
        };
        let query = match self.query.is_empty() {
            true => String::new(),
            false => format!("?{}", self.query),
        };
        let fragment = match self.fragment.is_empty() {
            true => String::new(),
            false => format!("#{}", self.fragment),
        };
        format!("{scheme}{path}{query}{fragment}")
    }
}

fn is_scheme_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
}

fn split_scheme(url: &str) -> (String, &str) {
    match url.find(':') {
        Some(index)
            if index > 0
                && url.starts_with(|first: char| first.is_ascii_alphabetic())
                && url[..index].chars().all(is_scheme_char) =>
        {
            (url[..index].to_ascii_lowercase(), &url[index + 1..])
        }
        _ => (String::new(), url),
    }
}

/// `urllib.parse.urlsplit(url)` for a URL whose netloc has no unbalanced IPv6 brackets, the
/// one input CPython raises `ValueError` for.
pub fn urlsplit(url: &str) -> ParsedUrl {
    let cleaned: String = url
        .trim_start_matches(|character: char| character <= ' ')
        .chars()
        .filter(|character| !UNSAFE_URL_CHARS.contains(character))
        .collect();
    let (scheme, rest) = split_scheme(&cleaned);
    let (netloc, rest) = match rest.strip_prefix("//") {
        Some(after) => {
            let end = after.find(['/', '?', '#']).unwrap_or(after.len());
            after.split_at(end)
        }
        None => ("", rest),
    };
    let (rest, fragment) = rest.split_once('#').unwrap_or((rest, ""));
    let (path, query) = rest.split_once('?').unwrap_or((rest, ""));
    ParsedUrl {
        scheme,
        netloc: netloc.to_string(),
        path: path.to_string(),
        params: String::new(),
        query: query.to_string(),
        fragment: fragment.to_string(),
    }
}

/// `urllib.parse.urlparse(url)`: [`urlsplit`], then `;params` split off the last path segment
/// for the schemes that carry them.
pub fn urlparse(url: &str) -> ParsedUrl {
    let split = urlsplit(url);
    if !USES_PARAMS.contains(&split.scheme.as_str()) || !split.path.contains(';') {
        return split;
    }
    let last_segment = split.path.rfind('/').unwrap_or(0);
    let separator = match split.path.contains('/') {
        true => split.path[last_segment..]
            .find(';')
            .map(|index| last_segment + index),
        false => split.path.find(';'),
    };
    let Some(separator) = separator else {
        return split;
    };
    ParsedUrl {
        path: split.path[..separator].to_string(),
        params: split.path[separator + 1..].to_string(),
        ..split
    }
}
