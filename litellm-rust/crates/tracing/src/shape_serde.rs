use crate::{ShapeSource, ShapeVisitor};
use serde::{
    Serialize, Serializer,
    ser::{
        Error as _, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant,
        SerializeTuple, SerializeTupleStruct, SerializeTupleVariant,
    },
};

pub(crate) struct Serialized<'a, T: ?Sized>(pub &'a T);
impl<T: Serialize + ?Sized> ShapeSource for Serialized<'_, T> {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        self.0
            .serialize(Fields {
                visitor,
                variant: None,
                key: None,
            })
            .is_ok()
    }
}

struct Fields<'a, 'b> {
    visitor: &'a mut ShapeVisitor<'b>,
    variant: Option<&'static str>,
    key: Option<String>,
}

struct Element<'a, T: ?Sized>(&'a T);
impl<T: Serialize + ?Sized> ShapeSource for Element<'_, T> {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        visitor.item(&Serialized(self.0))
    }
}
struct Field<'a, T: ?Sized>(&'a str, &'a T);
impl<T: Serialize + ?Sized> ShapeSource for Field<'_, T> {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        visitor.field(self.0, &Serialized(self.1))
    }
}
fn admitted(value: bool) -> Result<(), serde_json::Error> {
    if value {
        Ok(())
    } else {
        Err(serde_json::Error::custom("shape limit reached"))
    }
}

impl Fields<'_, '_> {
    fn item<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), serde_json::Error> {
        admitted(match self.variant {
            None => self.visitor.item(&Serialized(value)),
            Some(variant) => self.visitor.field(variant, &Element(value)),
        })
    }
    fn field<T: Serialize + ?Sized>(
        &mut self,
        key: &str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        admitted(match self.variant {
            None => self.visitor.field(key, &Serialized(value)),
            Some(variant) => self.visitor.field(variant, &Field(key, value)),
        })
    }
}

macro_rules! scalar {
    ($method:ident, $ty:ty) => {
        fn $method(self, _: $ty) -> Result<(), Self::Error> {
            Ok(())
        }
    };
}
impl<'a, 'b> Serializer for Fields<'a, 'b> {
    type Ok = ();
    type Error = serde_json::Error;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;
    scalar!(serialize_bool, bool);
    scalar!(serialize_i8, i8);
    scalar!(serialize_i16, i16);
    scalar!(serialize_i32, i32);
    scalar!(serialize_i64, i64);
    scalar!(serialize_i128, i128);
    scalar!(serialize_u8, u8);
    scalar!(serialize_u16, u16);
    scalar!(serialize_u32, u32);
    scalar!(serialize_u64, u64);
    scalar!(serialize_u128, u128);
    scalar!(serialize_f32, f32);
    scalar!(serialize_f64, f64);
    scalar!(serialize_char, char);
    scalar!(serialize_str, &str);
    fn serialize_bytes(mut self, value: &[u8]) -> Result<(), Self::Error> {
        for _ in value {
            self.item(&0u8)?;
        }
        Ok(())
    }
    fn serialize_none(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<(), Self::Error> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        _: &'static str,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        admitted(self.visitor.field(variant, &Serialized(value)))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, Self::Error> {
        admitted(self.visitor.field(variant, &serde_json::Value::Null))?;
        Ok(Self {
            variant: Some(variant),
            ..self
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, Self::Error> {
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, Self::Error> {
        admitted(self.visitor.field(variant, &serde_json::Value::Null))?;
        Ok(Self {
            variant: Some(variant),
            ..self
        })
    }
    fn collect_str<T: std::fmt::Display + ?Sized>(self, _: &T) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeMap for Fields<'_, '_> {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Self::Error> {
        let key = key.serialize(serde_json::value::Serializer)?;
        self.key = Some(match key {
            serde_json::Value::String(key) => key,
            key @ (serde_json::Value::Number(_) | serde_json::Value::Bool(_)) => key.to_string(),
            _ => return Err(serde_json::Error::custom("shape map key is not scalar")),
        });
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        let Some(key) = self.key.take() else {
            return Err(serde_json::Error::custom("shape map missing key"));
        };
        self.field(&key, value)
    }
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl SerializeStruct for Fields<'_, '_> {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}
impl SerializeStructVariant for Fields<'_, '_> {
    type Ok = ();
    type Error = serde_json::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.field(key, value)
    }
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}
macro_rules! sequence {
    ($trait:ident, $method:ident) => {
        impl $trait for Fields<'_, '_> {
            type Ok = ();
            type Error = serde_json::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
                self.item(value)
            }
            fn end(self) -> Result<(), Self::Error> {
                Ok(())
            }
        }
    };
}
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);
