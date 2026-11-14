use pyo3::exceptions::PyKeyError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyIterator, PyList,
    PyMapping, PySequence, PySet, PyString, PyTuple,
};
use pythonize::{PythonizeError, Result};
use serde::de::value::StrDeserializer;
use serde::de::{self, DeserializeOwned, DeserializeSeed, IntoDeserializer, Visitor};

pub(crate) fn depythonize<T>(input: &Bound<'_, PyAny>) -> Result<T>
where
    T: DeserializeOwned,
{
    T::deserialize(Depythonizer { input })
}

#[derive(Clone, Copy)]
struct Depythonizer<'a, 'py> {
    input: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> Depythonizer<'a, 'py> {
    fn leaf(self) -> pythonize::Depythonizer<'a, 'py> {
        pythonize::Depythonizer::from_object(self.input)
    }

    fn items(self) -> Result<Entries<'py>> {
        Entries::items(self.input.cast::<PyMapping>()?)
    }

    fn dataclass_fields(self) -> Result<Option<Entries<'py>>> {
        self.input
            .getattr_opt(intern!(self.input.py(), "__dataclass_fields__"))
            .ok()
            .flatten()
            .map(|fields| Entries::dataclass_fields(self.input, &fields))
            .transpose()
    }

    fn elements(self, expected_len: Option<usize>) -> Result<Option<Elements<'a, 'py>>> {
        let Ok(sequence) = self.input.cast::<PySequence>() else {
            return Ok(None);
        };
        let len = self.input.len()?;
        Ok(expected_len
            .is_none_or(|expected| expected == len)
            .then_some(Elements {
                sequence,
                index: 0,
                len,
            }))
    }

    fn tagged_variant(self) -> Result<Option<Variant<'py>>> {
        if self.input.is_instance_of::<PyString>() {
            return Ok(None);
        }
        let Ok(mapping) = self.input.cast::<PyMapping>() else {
            return Ok(None);
        };
        if mapping.len()? != 1 {
            return Ok(None);
        }
        let Ok(name) = mapping.keys()?.get_item(0)?.cast_into::<PyString>() else {
            return Ok(None);
        };
        let value = mapping.get_item(&name)?;
        Ok(Some(Variant { name, value }))
    }
}

macro_rules! delegate_to_pythonize {
    ($($method:ident)*) => {$(
        fn $method<V>(self, visitor: V) -> Result<V::Value>
        where
            V: Visitor<'de>,
        {
            self.leaf().$method(visitor)
        }
    )*};
}

impl<'de> de::Deserializer<'de> for Depythonizer<'_, '_> {
    type Error = PythonizeError;

    delegate_to_pythonize! {
        deserialize_bool deserialize_char deserialize_str deserialize_string
        deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i64 deserialize_i128
        deserialize_u8 deserialize_u16 deserialize_u32 deserialize_u64 deserialize_u128
        deserialize_f32 deserialize_f64 deserialize_bytes deserialize_byte_buf
        deserialize_unit deserialize_identifier deserialize_ignored_any
    }

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        let input = self.input;
        if input.is_none() || input.is_instance_of::<PyBool>() || input.is_instance_of::<PyInt>() {
            self.leaf().deserialize_any(visitor)
        } else if input.is_instance_of::<PyList>() || input.is_instance_of::<PyTuple>() {
            self.deserialize_tuple(input.len()?, visitor)
        } else if input.is_instance_of::<PyDict>() {
            visitor.visit_map(self.items()?)
        } else if input.is_instance_of::<PyString>()
            || input.is_instance_of::<PyBytes>()
            || input.is_instance_of::<PyByteArray>()
            || input.is_instance_of::<PyFloat>()
        {
            self.leaf().deserialize_any(visitor)
        } else if input.is_instance_of::<PyFrozenSet>() || input.is_instance_of::<PySet>() {
            self.deserialize_seq(visitor)
        } else if input.cast::<PySequence>().is_ok() {
            self.deserialize_tuple(input.len()?, visitor)
        } else if input.cast::<PyMapping>().is_ok() {
            self.deserialize_map(visitor)
        } else if let Some(fields) = self.dataclass_fields()? {
            visitor.visit_map(fields)
        } else {
            self.leaf().deserialize_any(visitor)
        }
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.input.is_none() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit_struct<V>(self, name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.leaf().deserialize_unit_struct(name, visitor)
    }

    fn deserialize_newtype_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if let Some(elements) = self.elements(None)? {
            return visitor.visit_seq(elements);
        }
        if self.input.is_instance_of::<PySet>() || self.input.is_instance_of::<PyFrozenSet>() {
            return visitor.visit_seq(SetElements {
                iterator: self.input.try_iter()?,
            });
        }
        self.leaf().deserialize_seq(visitor)
    }

    fn deserialize_tuple<V>(self, len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.elements(Some(len))? {
            Some(elements) => visitor.visit_seq(elements),
            None => self.leaf().deserialize_tuple(len, visitor),
        }
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_tuple(len, visitor)
    }

    fn deserialize_map<V>(self, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        if self.input.is_instance_of::<PyDict>() {
            return visitor.visit_map(self.items()?);
        }
        match self.dataclass_fields()? {
            Some(fields) => visitor.visit_map(fields),
            None => visitor.visit_map(self.items()?),
        }
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        match self.tagged_variant()? {
            Some(variant) => visitor.visit_enum(variant),
            None => self.leaf().deserialize_enum(name, variants, visitor),
        }
    }
}

struct Elements<'a, 'py> {
    sequence: &'a Bound<'py, PySequence>,
    index: usize,
    len: usize,
}

impl<'de> de::SeqAccess<'de> for Elements<'_, '_> {
    type Error = PythonizeError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        if self.index >= self.len {
            return Ok(None);
        }
        let element = self.sequence.get_item(self.index)?;
        self.index += 1;
        seed.deserialize(Depythonizer { input: &element }).map(Some)
    }
}

struct SetElements<'py> {
    iterator: Bound<'py, PyIterator>,
}

impl<'de> de::SeqAccess<'de> for SetElements<'_> {
    type Error = PythonizeError;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>>
    where
        T: DeserializeSeed<'de>,
    {
        self.iterator
            .next()
            .map(|element| seed.deserialize(Depythonizer { input: &element? }))
            .transpose()
    }
}

struct Entries<'py> {
    keys: Bound<'py, PyList>,
    values: Values<'py>,
    len: usize,
    key_index: usize,
    value_index: usize,
}

enum Values<'py> {
    Listed(Bound<'py, PyList>),
    DataclassAttributes(Bound<'py, PyDict>),
}

impl<'py> Entries<'py> {
    fn items(mapping: &Bound<'py, PyMapping>) -> Result<Self> {
        Ok(Self {
            keys: mapping.keys()?,
            values: Values::Listed(mapping.values()?),
            len: mapping.len()?,
            key_index: 0,
            value_index: 0,
        })
    }

    fn dataclass_fields(instance: &Bound<'py, PyAny>, fields: &Bound<'py, PyAny>) -> Result<Self> {
        let keys = fields.cast::<PyDict>()?.keys();
        let attributes = instance
            .getattr(intern!(instance.py(), "__dict__"))?
            .cast_into::<PyDict>()?;
        Ok(Self {
            len: keys.len(),
            keys,
            values: Values::DataclassAttributes(attributes),
            key_index: 0,
            value_index: 0,
        })
    }

    fn value(&self, index: usize) -> Result<Bound<'py, PyAny>> {
        match &self.values {
            Values::Listed(values) => Ok(values.get_item(index)?),
            Values::DataclassAttributes(attributes) => {
                let key = self.keys.get_item(index)?;
                attributes
                    .get_item(&key)?
                    .ok_or_else(|| PyKeyError::new_err(key.unbind()).into())
            }
        }
    }
}

impl<'de> de::MapAccess<'de> for Entries<'_> {
    type Error = PythonizeError;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>>
    where
        K: DeserializeSeed<'de>,
    {
        if self.key_index >= self.len {
            return Ok(None);
        }
        let key = self.keys.get_item(self.key_index)?;
        self.key_index += 1;
        seed.deserialize(Depythonizer { input: &key }).map(Some)
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value>
    where
        V: DeserializeSeed<'de>,
    {
        let value = self.value(self.value_index)?;
        self.value_index += 1;
        seed.deserialize(Depythonizer { input: &value })
    }
}

struct Variant<'py> {
    name: Bound<'py, PyString>,
    value: Bound<'py, PyAny>,
}

impl<'de> de::EnumAccess<'de> for Variant<'_> {
    type Error = PythonizeError;
    type Variant = Self;

    fn variant_seed<V>(self, seed: V) -> Result<(V::Value, Self)>
    where
        V: DeserializeSeed<'de>,
    {
        let name = self.name.to_cow()?;
        let tag: StrDeserializer<'_, PythonizeError> = name.as_ref().into_deserializer();
        Ok((seed.deserialize(tag)?, self))
    }
}

impl<'de> de::VariantAccess<'de> for Variant<'_> {
    type Error = PythonizeError;

    fn unit_variant(self) -> Result<()> {
        Ok(())
    }

    fn newtype_variant_seed<T>(self, seed: T) -> Result<T::Value>
    where
        T: DeserializeSeed<'de>,
    {
        seed.deserialize(Depythonizer { input: &self.value })
    }

    fn tuple_variant<V>(self, len: usize, visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        de::Deserializer::deserialize_tuple(Depythonizer { input: &self.value }, len, visitor)
    }

    fn struct_variant<V>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value>
    where
        V: Visitor<'de>,
    {
        visitor.visit_map(Depythonizer { input: &self.value }.items()?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ffi::CString;
    use std::fmt::Debug;

    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::{fixture, rstest};
    use serde::Deserialize;
    use serde::de::DeserializeOwned;
    use serde_json::{Value, json};

    use crate::{InitializedPython, from_py, from_py_argument, initialized_python};

    #[derive(Debug, Deserialize, PartialEq)]
    struct TextBlock {
        r#type: String,
        text: String,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Holder {
        block: TextBlock,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Wrapper(TextBlock);

    #[derive(Debug, Deserialize, PartialEq)]
    struct Twins(TextBlock, TextBlock);

    #[derive(Debug, Deserialize, PartialEq)]
    enum Tagged {
        Text(TextBlock),
        Pair(TextBlock, TextBlock),
        Wrapped { block: TextBlock },
    }

    #[derive(Debug, Default, Deserialize, PartialEq)]
    #[serde(default)]
    struct Request {
        block: Option<TextBlock>,
        listed: Vec<TextBlock>,
        paired: Option<(TextBlock,)>,
        keyed: BTreeMap<String, TextBlock>,
        held: Option<Holder>,
        wrapped: Option<Wrapper>,
        twins: Option<Twins>,
        tagged: Option<Tagged>,
    }

    type ToValue = fn(&Bound<'_, PyAny>) -> PyResult<Value>;
    type ErrorPair = fn(&Bound<'_, PyAny>) -> (String, String);

    #[fixture]
    fn namespace(#[from(initialized_python)] python: &InitializedPython) -> Py<PyDict> {
        python.attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"
import dataclasses
import types

class CatchAllDict(dict):
    def __getattr__(self, name):
        return self[name] if name in self else CatchAllDict()

@dataclasses.dataclass
class DataclassDict(dict):
    pass

@dataclasses.dataclass
class Block:
    type: str
    text: str

@dataclasses.dataclass
class Holder:
    block: object

@dataclasses.dataclass(frozen=True, eq=False)
class Pinned:
    block: object

def make(cls, **items):
    value = cls()
    value.update(items)
    return value

def block(cls):
    return make(cls, type='text', text='hi')

def without_text():
    value = Block(type='text', text='hi')
    del value.text
    return value
",
                Some(&namespace),
                None,
            )
            .unwrap();
            namespace.unbind()
        })
    }

    fn evaluate<'py>(py: Python<'py>, namespace: &Py<PyDict>, source: &str) -> Bound<'py, PyAny> {
        let source = CString::new(source).unwrap();
        py.eval(&source, Some(namespace.bind(py)), None).unwrap()
    }

    fn with_class(expression: &str, class: &str) -> String {
        expression
            .replace("BLOCK", "block(CLASS)")
            .replace("CLASS", class)
    }

    fn text_block() -> TextBlock {
        TextBlock {
            r#type: "text".to_owned(),
            text: "hi".to_owned(),
        }
    }

    fn text_block_json() -> Value {
        json!({"type": "text", "text": "hi"})
    }

    fn error_pair<T>(value: &Bound<'_, PyAny>) -> (String, String)
    where
        T: DeserializeOwned + Debug,
    {
        let ours = from_py::<T>(value).unwrap_err().to_string();
        let pythonize = PyErr::from(pythonize::depythonize::<T>(value).unwrap_err()).to_string();
        (ours, pythonize)
    }

    #[rstest]
    #[case::bare("BLOCK", text_block_json())]
    #[case::in_list("[BLOCK]", json!([text_block_json()]))]
    #[case::in_tuple("(BLOCK,)", json!([text_block_json()]))]
    #[case::in_dict("{'block': BLOCK}", json!({"block": text_block_json()}))]
    #[case::in_mapping(
        "types.MappingProxyType({'block': BLOCK})",
        json!({"block": text_block_json()})
    )]
    #[case::in_dataclass("Holder(block=BLOCK)", json!({"block": text_block_json()}))]
    #[case::in_set(
        "frozenset({Pinned(block=BLOCK)})",
        json!([{"block": text_block_json()}])
    )]
    fn a_dict_subclass_converts_by_its_items(
        namespace: Py<PyDict>,
        #[case] expression: &str,
        #[case] expected: Value,
        #[values("CatchAllDict", "DataclassDict")] class: &str,
        #[values(from_py::<Value>, from_py_argument::<Value>)] convert: ToValue,
    ) {
        Python::attach(|py| {
            let value = evaluate(py, &namespace, &with_class(expression, class));
            assert_eq!(convert(&value).unwrap(), expected);
        });
    }

    #[rstest]
    #[case::itself(
        "make(CLASS, block=BLOCK)",
        Request { block: Some(text_block()), ..Request::default() }
    )]
    #[case::optional(
        "{'block': BLOCK}",
        Request { block: Some(text_block()), ..Request::default() }
    )]
    #[case::in_list(
        "{'listed': [BLOCK]}",
        Request { listed: vec![text_block()], ..Request::default() }
    )]
    #[case::in_tuple(
        "{'paired': (BLOCK,)}",
        Request { paired: Some((text_block(),)), ..Request::default() }
    )]
    #[case::in_dict(
        "{'keyed': {'a': BLOCK}}",
        Request { keyed: BTreeMap::from([("a".to_owned(), text_block())]), ..Request::default() }
    )]
    #[case::in_dataclass(
        "{'held': Holder(block=BLOCK)}",
        Request { held: Some(Holder { block: text_block() }), ..Request::default() }
    )]
    #[case::newtype_variant(
        "{'tagged': {'Text': BLOCK}}",
        Request { tagged: Some(Tagged::Text(text_block())), ..Request::default() }
    )]
    #[case::newtype_struct(
        "{'wrapped': BLOCK}",
        Request { wrapped: Some(Wrapper(text_block())), ..Request::default() }
    )]
    #[case::tuple_struct(
        "{'twins': [BLOCK, BLOCK]}",
        Request { twins: Some(Twins(text_block(), text_block())), ..Request::default() }
    )]
    #[case::tuple_variant(
        "{'tagged': {'Pair': [BLOCK, BLOCK]}}",
        Request { tagged: Some(Tagged::Pair(text_block(), text_block())), ..Request::default() }
    )]
    #[case::struct_variant(
        "{'tagged': {'Wrapped': make(CLASS, block=BLOCK)}}",
        Request { tagged: Some(Tagged::Wrapped { block: text_block() }), ..Request::default() }
    )]
    fn a_dict_subclass_fills_a_typed_value(
        namespace: Py<PyDict>,
        #[case] expression: &str,
        #[case] expected: Request,
        #[values("CatchAllDict", "DataclassDict")] class: &str,
    ) {
        Python::attach(|py| {
            let value = evaluate(py, &namespace, &with_class(expression, class));
            assert_eq!(from_py::<Request>(&value).unwrap(), expected);
        });
    }

    #[rstest]
    #[case::dict("{'type': 'text', 'text': 'hi'}")]
    #[case::mapping("types.MappingProxyType({'type': 'text', 'text': 'hi'})")]
    #[case::dataclass("Block(type='text', text='hi')")]
    fn a_plain_mapping_or_dataclass_converts_by_its_fields(
        namespace: Py<PyDict>,
        #[case] expression: &str,
    ) {
        Python::attach(|py| {
            let value = evaluate(py, &namespace, expression);
            assert_eq!(from_py::<Value>(&value).unwrap(), text_block_json());
            assert_eq!(from_py::<TextBlock>(&value).unwrap(), text_block());
        });
    }

    #[rstest]
    #[case::unsupported_type("object()", error_pair::<Value>)]
    #[case::sequence_length("[1]", error_pair::<(i64, i64)>)]
    #[case::not_a_sequence("5", error_pair::<Vec<i64>>)]
    #[case::not_a_mapping("5", error_pair::<BTreeMap<String, i64>>)]
    #[case::enum_with_two_keys("{'Text': 1, 'Other': 2}", error_pair::<Tagged>)]
    #[case::enum_with_a_non_str_key("{1: 2}", error_pair::<Tagged>)]
    #[case::enum_of_another_type("5", error_pair::<Tagged>)]
    #[case::missing_dataclass_attribute("without_text()", error_pair::<Value>)]
    fn a_rejected_input_fails_as_pythonize_fails_it(
        namespace: Py<PyDict>,
        #[case] expression: &str,
        #[case] errors: ErrorPair,
    ) {
        Python::attach(|py| {
            let value = evaluate(py, &namespace, expression);
            let (ours, pythonize) = errors(&value);
            assert_eq!(ours, pythonize);
        });
    }
}
