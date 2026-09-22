use num_bigint::BigInt;

/// A Python value built only from literals: what `ast.literal_eval` can return.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Complex {
        re: f64,
        im: f64,
    },
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    /// Insertion-ordered, with Python's key equality already applied.
    Dict(Vec<(Value, Value)>),
    /// Literal order, with Python's member equality already applied.
    Set(Vec<Value>),
}

impl Value {
    /// Python's type name, as it appears in `TypeError` messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "NoneType",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Complex { .. } => "complex",
            Value::Str(_) => "str",
            Value::Bytes(_) => "bytes",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            Value::Set(_) => "set",
        }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value.into())
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Str(value.to_owned())
    }
}
