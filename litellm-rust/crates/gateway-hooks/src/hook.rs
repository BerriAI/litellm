//! Hooks run on the request path, in the order the config lists them. Each declares the facts it
//! reads and the parts of the request it may change; several hooks may change the same part, and
//! the later one in the config wins.

use std::{
    borrow::Cow,
    ops::{BitOr, BitOrAssign},
    sync::Arc,
};

use futures_util::future::BoxFuture;
use http::{HeaderName, HeaderValue};
use serde_json::{Map, Value};

use crate::facts::{Annotations, Facts, Record, RequestFacts};

/// The parts of a request a hook may change.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Targets(u8);

impl Targets {
    pub const NONE: Self = Self(0);
    /// Turn the request away before it is sent.
    pub const ADMISSION: Self = Self(1 << 0);
    /// Headers sent to the provider.
    pub const UPSTREAM_HEADERS: Self = Self(1 << 1);
    /// Top-level fields of the body sent to the provider.
    pub const UPSTREAM_BODY: Self = Self(1 << 2);
    /// Headers returned to the client.
    pub const RESPONSE_HEADERS: Self = Self(1 << 3);
    /// Values handed to observers with the request's record.
    pub const ANNOTATIONS: Self = Self(1 << 4);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl BitOr for Targets {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Targets {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

/// Matches header names, case-insensitively.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeaderPattern {
    Exact(HeaderName),
    /// Every header whose name starts with this lowercase prefix.
    Prefix(String),
}

impl HeaderPattern {
    /// `x-foo-*` is a prefix pattern; anything else is one header.
    pub fn parse(pattern: &str) -> Result<Self, http::header::InvalidHeaderName> {
        match pattern.strip_suffix('*') {
            Some(prefix) => Ok(Self::Prefix(prefix.to_ascii_lowercase())),
            None => HeaderName::from_bytes(pattern.as_bytes()).map(Self::Exact),
        }
    }

    pub fn matches(&self, name: &str) -> bool {
        match self {
            Self::Exact(exact) => exact.as_str().eq_ignore_ascii_case(name),
            Self::Prefix(prefix) => name
                .get(..prefix.len())
                .is_some_and(|start| start.eq_ignore_ascii_case(prefix)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum HeaderOp {
    Set(HeaderName, HeaderValue),
    Remove(HeaderPattern),
}

/// Why a hook turned a request away. The client gets `status`, `message` and `headers`.
#[derive(Clone, Debug, PartialEq)]
pub struct Refusal {
    pub status: u16,
    /// The proxy's error `type`, such as `budget_exceeded`.
    pub kind: Cow<'static, str>,
    pub message: String,
    pub headers: Vec<(HeaderName, HeaderValue)>,
}

impl Refusal {
    pub fn new(
        status: u16,
        kind: impl Into<Cow<'static, str>>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            kind: kind.into(),
            message: message.into(),
            headers: Vec::new(),
        }
    }

    pub fn with_header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.push((name, value));
        self
    }
}

/// The changes hooks asked for, kept in the order they were made. A hook may only make the
/// changes its [`Hook::writes`] declares; anything else is dropped with a warning.
#[derive(Debug, Default)]
pub struct RequestEdit {
    allowed: Targets,
    writer: &'static str,
    header_ops: Vec<HeaderOp>,
    body_fields: Vec<(String, Value)>,
    response_headers: Vec<(HeaderName, HeaderValue)>,
    annotations: Annotations,
}

impl RequestEdit {
    pub(crate) fn scope(&mut self, writer: &'static str, allowed: Targets) {
        self.writer = writer;
        self.allowed = allowed;
    }

    fn may(&self, target: Targets) -> bool {
        let allowed = self.allowed.contains(target);
        if !allowed {
            tracing::warn!(
                hook = self.writer,
                ?target,
                "hook wrote a target it does not declare; ignored"
            );
        }
        allowed
    }

    pub fn set_upstream_header(&mut self, name: HeaderName, value: HeaderValue) {
        if self.may(Targets::UPSTREAM_HEADERS) {
            self.header_ops.push(HeaderOp::Set(name, value));
        }
    }

    pub fn remove_upstream_headers(&mut self, pattern: HeaderPattern) {
        if self.may(Targets::UPSTREAM_HEADERS) {
            self.header_ops.push(HeaderOp::Remove(pattern));
        }
    }

    /// Sets a top-level field of the upstream body, replacing what the client sent.
    pub fn set_upstream_body_field(&mut self, field: impl Into<String>, value: Value) {
        if self.may(Targets::UPSTREAM_BODY) {
            self.body_fields.push((field.into(), value));
        }
    }

    pub fn add_response_header(&mut self, name: HeaderName, value: HeaderValue) {
        if self.may(Targets::RESPONSE_HEADERS) {
            self.response_headers.push((name, value));
        }
    }

    pub fn annotate(&mut self, key: &'static str, value: Value) {
        if self.may(Targets::ANNOTATIONS) {
            self.annotations.set(key, value);
        }
    }

    /// Whether any change reaches the provider.
    pub fn changes_upstream(&self) -> bool {
        !self.header_ops.is_empty() || !self.body_fields.is_empty()
    }

    /// Applies the header changes in order to the headers about to be sent.
    pub fn apply_upstream_headers(&self, headers: &mut Vec<(String, String)>) {
        for op in &self.header_ops {
            match op {
                HeaderOp::Set(name, value) => {
                    let Ok(value) = value.to_str() else { continue };
                    headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name.as_str()));
                    headers.push((name.as_str().to_owned(), value.to_owned()));
                }
                HeaderOp::Remove(pattern) => headers.retain(|(name, _)| !pattern.matches(name)),
            }
        }
    }

    /// Applies the body changes in order to the body about to be sent.
    pub fn apply_upstream_body(&self, body: &mut Map<String, Value>) {
        for (field, value) in &self.body_fields {
            body.insert(field.clone(), value.clone());
        }
    }

    pub fn response_headers(&self) -> &[(HeaderName, HeaderValue)] {
        &self.response_headers
    }

    pub(crate) fn take_annotations(&mut self) -> Annotations {
        std::mem::take(&mut self.annotations)
    }
}

/// Work a hook left to do once the response ends, such as returning an unused reservation.
pub trait Settle: Send + Sync + 'static {
    fn settle(self: Box<Self>, record: &Record);
}

/// An on-path hook.
pub trait Hook: Send + Sync + 'static {
    /// The name the config uses for this hook.
    fn name(&self) -> &'static str;

    /// The facts this hook reads.
    fn reads(&self) -> Facts;

    /// The parts of the request this hook may change.
    fn writes(&self) -> Targets;

    /// Runs before the request is routed. Returning a [`Settle`] asks to be told how the request
    /// ended.
    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>>;
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::exact("x-a", "X-A", true)]
    #[case::exact_other("x-a", "x-ab", false)]
    #[case::prefix("x-internal-*", "X-Internal-Trace", true)]
    #[case::prefix_short("x-internal-*", "x-intern", false)]
    fn header_patterns_ignore_case(#[case] pattern: &str, #[case] name: &str, #[case] hit: bool) {
        assert_eq!(HeaderPattern::parse(pattern).unwrap().matches(name), hit);
    }

    #[rstest]
    fn header_changes_apply_in_order() {
        let mut edit = RequestEdit::default();
        edit.scope("first", Targets::UPSTREAM_HEADERS);
        edit.set_upstream_header(
            HeaderName::from_static("x-a"),
            HeaderValue::from_static("1"),
        );
        edit.scope("second", Targets::UPSTREAM_HEADERS);
        edit.remove_upstream_headers(HeaderPattern::parse("x-*").unwrap());
        edit.set_upstream_header(
            HeaderName::from_static("x-b"),
            HeaderValue::from_static("2"),
        );

        let mut sent = headers(&[("X-A", "client"), ("content-type", "application/json")]);
        edit.apply_upstream_headers(&mut sent);
        assert_eq!(
            sent,
            headers(&[("content-type", "application/json"), ("x-b", "2")])
        );
    }

    #[rstest]
    fn later_hooks_win_on_the_same_body_field() {
        let mut edit = RequestEdit::default();
        edit.scope("first", Targets::UPSTREAM_BODY);
        edit.set_upstream_body_field("metadata", Value::from("a"));
        edit.scope("second", Targets::UPSTREAM_BODY);
        edit.set_upstream_body_field("metadata", Value::from("b"));

        let serde_json::Value::Object(mut body) =
            serde_json::json!({"model": "m", "metadata": "client"})
        else {
            unreachable!()
        };
        edit.apply_upstream_body(&mut body);
        assert_eq!(
            Value::Object(body),
            serde_json::json!({"model": "m", "metadata": "b"})
        );
    }

    #[rstest]
    fn undeclared_writes_are_dropped() {
        let mut edit = RequestEdit::default();
        edit.scope("reader", Targets::ANNOTATIONS);
        edit.set_upstream_body_field("metadata", Value::Null);
        edit.add_response_header(
            HeaderName::from_static("x-a"),
            HeaderValue::from_static("1"),
        );
        edit.annotate("a", Value::from(1));
        assert!(!edit.changes_upstream());
        assert!(edit.response_headers().is_empty());
        assert_eq!(edit.take_annotations().get("a"), Some(&Value::from(1)));
    }
}
