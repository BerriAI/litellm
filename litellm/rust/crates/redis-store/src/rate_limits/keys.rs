/// A rate-limit descriptor, `{descriptor:value}`, the hash tag every key of its window shares.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scope {
    pub descriptor: String,
    pub value: String,
}

impl Scope {
    pub(super) fn key(&self, suffix: &str) -> String {
        format!("{{{}:{}}}:{suffix}", self.descriptor, self.value)
    }

    pub(super) fn window_key(&self) -> String {
        self.key(WINDOW)
    }

    pub(super) fn gauge_key(&self) -> String {
        self.key(GAUGE)
    }
}

const WINDOW: &str = "window";
const GAUGE: &str = "max_parallel_requests";

/// A counter's key suffix, which is also how the Python limiter names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::EnumString, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "requests")]
    Requests,
    #[strum(serialize = "tokens")]
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Counter {
    pub scope: Scope,
    pub kind: Kind,
}

impl Counter {
    pub(super) fn key(&self) -> String {
        self.scope.key(self.kind.into())
    }
}

/// One value of the limiter's keyspace, as both tiers address it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Window(Scope),
    Counter(Counter),
    Gauge(Scope),
}

impl Entry {
    /// The entry the Python limiter names by its key suffix: `window`, `max_parallel_requests`, or a
    /// counter's `Kind`.
    pub fn new(suffix: &str, scope: Scope) -> Result<Self, strum::ParseError> {
        match suffix {
            WINDOW => Ok(Self::Window(scope)),
            GAUGE => Ok(Self::Gauge(scope)),
            kind => kind
                .parse()
                .map(|kind| Self::Counter(Counter { scope, kind })),
        }
    }

    pub(super) fn key(&self) -> String {
        match self {
            Self::Window(scope) => scope.window_key(),
            Self::Counter(counter) => counter.key(),
            Self::Gauge(scope) => scope.gauge_key(),
        }
    }
}
