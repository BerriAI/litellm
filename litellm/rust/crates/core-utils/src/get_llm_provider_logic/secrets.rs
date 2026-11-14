use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deferred;

pub trait SecretReader {
    fn read(&self, name: &str) -> Result<Option<String>, Deferred>;
}

pub struct ProcessEnvironment;

impl SecretReader for ProcessEnvironment {
    fn read(&self, name: &str) -> Result<Option<String>, Deferred> {
        match std::env::var_os(name) {
            None => Ok(None),
            Some(value) => value.into_string().map(Some).map_err(|_| Deferred),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Text(String),
    Flag(bool),
}

impl Value {
    pub fn truthy(&self) -> bool {
        match self {
            Self::Text(text) => !text.is_empty(),
            Self::Flag(flag) => *flag,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => formatter.write_str(text),
            Self::Flag(true) => formatter.write_str("True"),
            Self::Flag(false) => formatter.write_str("False"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    Given(Option<&'a str>),
    SecretStr(&'static str),
    Secret(&'static str),
    Env(&'static str),
    Literal(&'a str),
}

fn python_strip(value: &str) -> &str {
    value.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn flag(value: &str) -> Option<bool> {
    match python_strip(value).to_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

pub struct Secrets<'a> {
    reader: &'a dyn SecretReader,
    manager: bool,
}

impl<'a> Secrets<'a> {
    pub fn new(reader: &'a dyn SecretReader, manager: bool) -> Self {
        Self { reader, manager }
    }

    fn raw(&self, name: &str) -> Result<Option<String>, Deferred> {
        if self.manager {
            return Err(Deferred);
        }
        self.reader.read(name)
    }

    pub fn secret(&self, name: &str) -> Result<Option<Value>, Deferred> {
        Ok(self.raw(name)?.map(|value| match flag(&value) {
            Some(flag) => Value::Flag(flag),
            None => Value::Text(value),
        }))
    }

    pub fn secret_str(&self, name: &str) -> Result<Option<String>, Deferred> {
        Ok(match self.secret(name)? {
            Some(Value::Text(text)) => Some(text),
            _ => None,
        })
    }

    pub fn secret_bool(&self, name: &str) -> Result<Option<bool>, Deferred> {
        Ok(match self.secret(name)? {
            Some(Value::Flag(flag)) => Some(flag),
            _ => None,
        })
    }

    fn source(&self, source: Source<'_>) -> Result<Option<Value>, Deferred> {
        Ok(match source {
            Source::Given(value) => value.map(|value| Value::Text(value.to_owned())),
            Source::SecretStr(name) => self.secret_str(name)?.map(Value::Text),
            Source::Secret(name) => self.secret(name)?,
            Source::Env(name) => self.raw(name)?.map(Value::Text),
            Source::Literal(value) => Some(Value::Text(value.to_owned())),
        })
    }

    pub fn first(&self, sources: &[Source<'_>]) -> Result<Option<Value>, Deferred> {
        let mut last = None;
        for source in sources {
            last = self.source(*source)?;
            if last.as_ref().is_some_and(Value::truthy) {
                break;
            }
        }
        Ok(last)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;

    use super::*;

    struct Fixed(HashMap<&'static str, &'static str>);

    impl SecretReader for Fixed {
        fn read(&self, name: &str) -> Result<Option<String>, Deferred> {
            Ok(self.0.get(name).map(|value| (*value).to_owned()))
        }
    }

    #[rstest]
    #[case::text("value", Some(Value::Text("value".into())))]
    #[case::padded_true(" TRUE\t", Some(Value::Flag(true)))]
    #[case::false_flag("false", Some(Value::Flag(false)))]
    #[case::empty("", Some(Value::Text(String::new())))]
    fn secret_converts_boolean_strings_like_python(
        #[case] raw: &'static str,
        #[case] expected: Option<Value>,
    ) {
        let reader = Fixed(HashMap::from([("NAME", raw)]));
        assert_eq!(Secrets::new(&reader, false).secret("NAME"), Ok(expected));
    }

    #[rstest]
    #[case::first_truthy(&[Source::Given(Some("")), Source::Literal("x")], Some("x"))]
    #[case::last_falsy(&[Source::Given(None), Source::Given(Some(""))], Some(""))]
    #[case::all_missing(&[Source::Given(None), Source::SecretStr("MISSING")], None)]
    fn first_follows_python_or_semantics(
        #[case] sources: &[Source<'static>],
        #[case] expected: Option<&str>,
    ) {
        let reader = Fixed(HashMap::new());
        assert_eq!(
            Secrets::new(&reader, false).first(sources),
            Ok(expected.map(|value| Value::Text(value.to_owned())))
        );
    }

    #[rstest]
    fn a_configured_secret_manager_defers_every_read() {
        let reader = Fixed(HashMap::from([("NAME", "value")]));
        assert_eq!(Secrets::new(&reader, true).secret("NAME"), Err(Deferred));
    }
}
