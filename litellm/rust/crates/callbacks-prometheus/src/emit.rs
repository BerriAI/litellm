use litellm_callbacks::CallbackError;
use serde_json::Value;

use crate::{
    labels::{LabelContext, Settings, label_strings},
    pyfmt::{is_py_int, is_py_number, py_big_int, py_number, py_type_name},
    store::{FamilyId, Op, Store},
};

pub fn failed(message: impl Into<String>) -> CallbackError {
    CallbackError::Failed(message.into())
}

pub fn raised(error_type: &str, text: &str) -> CallbackError {
    failed(format!("{error_type}: {text}"))
}

pub fn named_error(message: &str) -> Option<(&str, &str)> {
    let (name, text) = message.split_once(": ")?;
    let mut chars = name.chars();
    let named = chars.next().is_some_and(|first| first.is_ascii_uppercase())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_');
    named.then_some((name, text))
}

pub fn not_comparable(operator: &str, value: &Value) -> CallbackError {
    failed(format!(
        "TypeError: '{operator}' not supported between instances of '{}' and 'int'",
        py_type_name(value)
    ))
}

pub fn not_subtractable(end: bool, start: bool) -> CallbackError {
    let name = |present: bool| {
        if present {
            "datetime.datetime"
        } else {
            "NoneType"
        }
    };
    failed(format!(
        "TypeError: unsupported operand type(s) for -: '{}' and '{}'",
        name(end),
        name(start)
    ))
}

pub fn py_float(value: &Value) -> Result<f64, CallbackError> {
    match value {
        _ if is_py_number(value) => {
            py_number(value).ok_or_else(|| failed("float() of a number failed"))
        }
        Value::Bool(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => text
            .trim()
            .parse::<f64>()
            .map_err(|_| failed(format!("could not convert string to float: '{text}'"))),
        other => Err(failed(format!(
            "TypeError: float() argument must be a string or a real number, not '{}'",
            py_type_name(other)
        ))),
    }
}

pub fn py_int_or_bool(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(flag) => Some(if *flag { 1.0 } else { 0.0 }),
        _ if is_py_int(value) => py_number(value),
        _ => None,
    }
}

pub fn py_int(value: &Value) -> Option<i128> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .map(i128::from)
            .or_else(|| number.as_u64().map(i128::from)),
        other => py_big_int(other),
    }
}

pub fn py_positive_number(value: &Value) -> Option<f64> {
    match value {
        _ if is_py_number(value) => py_number(value).filter(|number| *number > 0.0),
        _ => None,
    }
}

pub struct Labels {
    pub names: Vec<String>,
    pub values: Vec<Option<String>>,
}

impl Labels {
    pub fn set(&mut self, name: &str, value: Option<String>) {
        if let Some(position) = self.names.iter().position(|label| label == name) {
            self.values[position] = value;
        }
    }
}

pub struct Emitter<'a> {
    pub store: &'a Store,
    pub settings: &'a Settings,
}

impl Emitter<'_> {
    pub fn labels(&self, name: &str, context: &LabelContext) -> Labels {
        let names = self.store.metric_labels(name);
        let values = context.labels(&names);
        Labels { names, values }
    }

    fn kwargs_values(&self, id: FamilyId, labels: &Labels) -> Result<Vec<String>, CallbackError> {
        let family = self.store.labelnames(id);
        let lookup = |name: &String| {
            labels
                .names
                .iter()
                .position(|label| label == name)
                .map(|position| labels.values[position].clone())
        };
        if self.store.is_wrapped(id) {
            return family
                .iter()
                .map(|name| lookup(name).ok_or_else(|| failed(format!("KeyError: '{name}'"))))
                .collect::<Result<Vec<_>, _>>()
                .map(|values| label_strings(&values));
        }
        if family.is_empty() {
            return Err(failed(
                "No label names were set when constructing the metric",
            ));
        }
        let mut wanted = family.clone();
        let mut given = labels.names.clone();
        wanted.sort();
        given.sort();
        if wanted != given {
            return Err(failed("Incorrect label names"));
        }
        Ok(label_strings(
            &family
                .iter()
                .map(|name| lookup(name).flatten())
                .collect::<Vec<_>>(),
        ))
    }

    pub fn record(
        &self,
        metric: &str,
        labels: &Labels,
        op: Op,
        track_end_user: bool,
    ) -> Result<(), CallbackError> {
        self.record_with(metric, labels, || Ok(op), track_end_user)
    }

    pub fn record_with<E: From<CallbackError>>(
        &self,
        metric: &str,
        labels: &Labels,
        op: impl FnOnce() -> Result<Op, E>,
        track_end_user: bool,
    ) -> Result<(), E> {
        let Some(id) = self.store.family(metric) else {
            return Ok(());
        };
        let values = self.kwargs_values(id, labels)?;
        let target = self
            .store
            .labels(id, &values)
            .map_err(|error| failed(error.to_string()))?;
        self.store
            .apply(id, &target, op()?)
            .map_err(|error| failed(error.to_string()))?;
        if track_end_user {
            self.track_end_user(id, labels);
        }
        Ok(())
    }

    fn track_end_user(&self, id: FamilyId, labels: &Labels) {
        let Some(position) = labels.names.iter().position(|name| name == "end_user") else {
            return;
        };
        if labels.values[position].is_none() {
            return;
        }
        self.store
            .track_end_user(id, &labels.values, &self.settings.end_user_limits());
    }

    pub fn inc(
        &self,
        name: &str,
        context: &LabelContext,
        amount: f64,
    ) -> Result<(), CallbackError> {
        self.record(name, &self.labels(name, context), Op::Inc(amount), true)
    }

    pub fn observe(
        &self,
        name: &str,
        context: &LabelContext,
        amount: f64,
    ) -> Result<(), CallbackError> {
        self.record(name, &self.labels(name, context), Op::Observe(amount), true)
    }

    pub fn remove(&self, name: &str, labels: &Labels) {
        if let Some(id) = self.store.family(name) {
            self.store.remove_end_user_series(id, &labels.values);
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::success::exception_text;

    #[rstest]
    #[case::int(json!(3), Ok(3.0))]
    #[case::bool(json!(true), Ok(1.0))]
    #[case::padded_text(json!(" 2.5 "), Ok(2.5))]
    #[case::bad_text(json!("ten"), Err("could not convert string to float: 'ten'"))]
    #[case::list(json!([1]), Err("float() argument must be a string or a real number, not 'list'"))]
    #[case::none(json!(null), Err("float() argument must be a string or a real number, not 'NoneType'"))]
    fn float_follows_the_python_builtin(#[case] value: Value, #[case] expected: Result<f64, &str>) {
        assert_eq!(
            py_float(&value).map_err(|error| exception_text(&error)),
            expected.map_err(str::to_owned)
        );
    }

    #[rstest]
    #[case::comparison(not_comparable(">=", &json!("slow")), "'>=' not supported between instances of 'str' and 'int'")]
    #[case::missing_end(
        not_subtractable(false, true),
        "unsupported operand type(s) for -: 'NoneType' and 'datetime.datetime'"
    )]
    #[case::missing_start(
        not_subtractable(true, false),
        "unsupported operand type(s) for -: 'datetime.datetime' and 'NoneType'"
    )]
    fn type_errors_carry_the_python_text(#[case] error: CallbackError, #[case] expected: &str) {
        assert_eq!(exception_text(&error), expected);
    }
}
