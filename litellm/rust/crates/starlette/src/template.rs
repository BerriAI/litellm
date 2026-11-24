use std::collections::BTreeSet;

/// A starlette path convertor, by the name a template spells it with. `{name}` is `Str`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString)]
pub enum Convertor {
    #[strum(serialize = "str")]
    Str,
    #[strum(serialize = "path")]
    Path,
    #[strum(serialize = "int")]
    Int,
    #[strum(serialize = "float")]
    Float,
    #[strum(serialize = "uuid")]
    Uuid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Param {
    pub convertor: Convertor,
    /// The template has a `:` anywhere after this parameter, as the Google
    /// `{model:path}:generateContent` routes do.
    pub colon_after: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplatePart {
    Literal(Box<str>),
    Param(Param),
}

/// A route template such as `/threads/{thread_id}` or a pass-through `path`. Every string parses:
/// a convertor starlette does not know reads as `Str`, and a host template such as
/// `chat_completion:80` drops its port.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(from = "String")]
pub struct RouteTemplate(Box<[TemplatePart]>);

impl From<String> for RouteTemplate {
    fn from(template: String) -> Self {
        Self::parse(&template)
    }
}

impl RouteTemplate {
    pub fn parse(template: &str) -> Self {
        let is_host = !template.starts_with('/');
        let mut parts = Vec::new();
        let mut rest = template;
        while let Some((before, convertor, after)) = next_param(rest) {
            parts.push(TemplatePart::Literal(before.into()));
            parts.push(TemplatePart::Param(Param {
                convertor: convertor
                    .and_then(|name| name.parse().ok())
                    .unwrap_or(Convertor::Str),
                colon_after: after.contains(':'),
            }));
            rest = after;
        }
        let tail = if is_host {
            rest.split(':').next().unwrap_or_default()
        } else {
            rest
        };
        parts.push(TemplatePart::Literal(tail.into()));
        Self(
            parts
                .into_iter()
                .filter(|part| !matches!(part, TemplatePart::Literal(text) if text.is_empty()))
                .collect(),
        )
    }

    pub fn parts(&self) -> &[TemplatePart] {
        &self.0
    }

    /// The route matches the regex `compile_path` builds from the template. Each convertor takes
    /// what its starlette regex takes: `str` one or more of anything but `/`, `path` zero or more
    /// of anything but a newline, `int` digits, `float` digits with an optional fraction, and
    /// `uuid` 32 hex digits with optional dashes between the groups.
    pub fn matches(&self, route: &str) -> bool {
        self.matches_with(route, |param, rest| convertor_ends(param.convertor, rest))
    }

    /// The route matches with literal text compared exactly and each parameter taking any length
    /// `param_ends` yields for the rest of the route. Like `re.match` against a pattern ending in
    /// `$`, a trailing newline is allowed.
    pub fn matches_with(
        &self,
        route: &str,
        param_ends: impl for<'r> Fn(Param, &'r str) -> Box<dyn Iterator<Item = usize> + 'r>,
    ) -> bool {
        walk(&self.0, route, &param_ends)
    }
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The next `{name}` or `{name:convertor}`, as starlette's `PARAM_REGEX` finds it: the text before
/// it, its convertor and the text after it.
fn next_param(text: &str) -> Option<(&str, Option<&str>, &str)> {
    text.match_indices('{').find_map(|(start, _)| {
        let end = start + text[start..].find('}')?;
        let inner = &text[start + 1..end];
        let (name, convertor) = match inner.split_once(':') {
            Some((name, convertor)) => (name, Some(convertor)),
            None => (inner, None),
        };
        (is_identifier(name) && convertor.is_none_or(is_identifier))
            .then(|| (&text[..start], convertor, &text[end + 1..]))
    })
}

fn is_end(rest: &str) -> bool {
    rest.is_empty() || rest == "\n"
}

/// Literals never branch, so they are stripped in place. From the first parameter on, the walk
/// keeps the set of offsets the parts so far can end at instead of trying each split in turn,
/// which keeps it at parts x route length x choices per parameter however many parameters a
/// template has.
fn walk(
    parts: &[TemplatePart],
    route: &str,
    param_ends: &impl for<'r> Fn(Param, &'r str) -> Box<dyn Iterator<Item = usize> + 'r>,
) -> bool {
    match parts.split_first() {
        None => is_end(route),
        Some((TemplatePart::Literal(literal), rest)) => route
            .strip_prefix(&**literal)
            .is_some_and(|after| walk(rest, after, param_ends)),
        Some((TemplatePart::Param(_), _)) => parts
            .iter()
            .try_fold(BTreeSet::from([0]), |starts, part| {
                let ends: BTreeSet<usize> = match part {
                    TemplatePart::Literal(literal) => starts
                        .into_iter()
                        .filter(|&start| route[start..].starts_with(&**literal))
                        .map(|start| start + literal.len())
                        .collect(),
                    TemplatePart::Param(param) => starts
                        .into_iter()
                        .flat_map(|start| {
                            param_ends(*param, &route[start..]).map(move |len| start + len)
                        })
                        .collect(),
                };
                (!ends.is_empty()).then_some(ends)
            })
            .is_some_and(|ends| ends.into_iter().any(|end| is_end(&route[end..]))),
    }
}

/// Every length of `route`'s start the convertor's regex can match.
fn convertor_ends(convertor: Convertor, route: &str) -> Box<dyn Iterator<Item = usize> + '_> {
    let boundaries = move |limit: usize, min: usize| {
        route
            .char_indices()
            .map(|(index, _)| index)
            .chain([route.len()])
            .filter(move |index| *index >= min && *index <= limit)
    };
    let digits = |text: &str| text.bytes().take_while(u8::is_ascii_digit).count();
    match convertor {
        Convertor::Str => Box::new(boundaries(route.find('/').unwrap_or(route.len()), 1)),
        Convertor::Path => Box::new(boundaries(route.find('\n').unwrap_or(route.len()), 0)),
        Convertor::Int => Box::new(1..=digits(route)),
        Convertor::Float => {
            let whole = digits(route);
            let fraction = route[whole..]
                .strip_prefix('.')
                .map(digits)
                .unwrap_or_default();
            let fractions = (whole > 0)
                .then_some(1..=fraction)
                .into_iter()
                .flatten()
                .map(move |digits| whole + 1 + digits);
            Box::new((1..=whole).chain(fractions))
        }
        Convertor::Uuid => Box::new(uuid_len(route).into_iter()),
    }
}

/// `[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}`. A dash can
/// never be a hex digit, so the match length is unique.
fn uuid_len(route: &str) -> Option<usize> {
    let bytes = route.as_bytes();
    [8, 4, 4, 4, 12]
        .iter()
        .enumerate()
        .try_fold(0, |at, (group, &width)| {
            let at = match (group > 0, bytes.get(at)) {
                (true, Some(b'-')) => at + 1,
                _ => at,
            };
            bytes
                .get(at..at + width)
                .filter(|hex| hex.iter().all(u8::is_ascii_hexdigit))
                .map(|_| at + width)
        })
}
