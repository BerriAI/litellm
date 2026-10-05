//! `ast.literal_eval(text)`, as a single-pass recursive-descent parser.
//!
//! Tokens follow CPython's tokenizer (string prefixes, escapes, implicit concatenation,
//! numeric underscores and radixes, comments, line continuations). Expressions follow
//! `ast.literal_eval`'s evaluator:
//!
//! - one unary `+`/`-`, applied only to a numeric constant (`-(1)` is fine, `--1` is not)
//! - `a + b` / `a - b` only as a signed real plus or minus a complex constant, with 3.14's
//!   mixed-mode rules (`1 - 0j` is `(1-0j)`)
//! - parentheses group without making a tuple; `set()` is the only call
//! - dict keys and set members are deduplicated with Python equality (`1 == 1.0 == True`)
//!
//! Not supported, each pinned by a fixture: `\N{NAME}` escapes, `...`, and escapes that
//! produce lone surrogates.

use std::collections::{HashMap, hash_map::Entry};

use num_bigint::BigInt;
use num_traits::{FromPrimitive, ToPrimitive};

use crate::{Error, MAX_DEPTH, Value};

pub fn literal_eval(text: &str) -> Result<Value, Error> {
    let mut parser = Parser {
        bytes: text.trim_start_matches([' ', '\t']).as_bytes(),
        offset: text.len() - text.trim_start_matches([' ', '\t']).len(),
        pos: 0,
        depth: 0,
        brackets: 0,
    };
    parser.skip_leading_lines()?;
    let value = parser.top_level()?.value;
    parser.skip_trivia(true);
    if parser.pos != parser.bytes.len() {
        return Err(parser.error());
    }
    Ok(value)
}

/// How a parsed term may take part in `+`/`-`, per `ast.literal_eval`'s `_convert_num`
/// (`Constant`) and `_convert_signed_num` (`Signed`). `Other` is any other node.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Constant,
    Signed,
    Other,
}

struct Term {
    value: Value,
    kind: Kind,
}

impl Term {
    fn other(value: Value) -> Self {
        Self {
            value,
            kind: Kind::Other,
        }
    }

    fn is_number(&self) -> bool {
        matches!(
            self.value,
            Value::Int(_) | Value::Float(_) | Value::Complex { .. }
        )
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    /// Bytes stripped before `bytes` starts, so errors report offsets into the input.
    offset: usize,
    pos: usize,
    depth: usize,
    /// Open brackets: newlines are insignificant only inside them.
    brackets: usize,
}

impl Parser<'_> {
    fn error(&self) -> Error {
        Error::InvalidLiteral(self.offset + self.pos)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<u8> {
        self.bytes.get(self.pos + ahead).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), Error> {
        if self.peek() != Some(byte) {
            return Err(self.error());
        }
        self.pos += 1;
        Ok(())
    }

    fn enter(&mut self) -> Result<(), Error> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Error::TooDeep);
        }
        Ok(())
    }

    /// Whitespace, comments, and backslash continuations; newlines too when `newlines`.
    fn skip_trivia(&mut self, newlines: bool) {
        while let Some(byte) = self.peek() {
            match byte {
                b' ' | b'\t' | b'\x0c' => self.pos += 1,
                b'#' => {
                    while !matches!(self.peek(), None | Some(b'\n' | b'\r')) {
                        self.pos += 1;
                    }
                }
                // A continuation joins two lines; one that ends the input is an EOF error.
                b'\\' if matches!(self.peek_at(1), Some(b'\n' | b'\r')) => {
                    let len = match (self.peek_at(1), self.peek_at(2)) {
                        (Some(b'\r'), Some(b'\n')) => 3,
                        _ => 2,
                    };
                    if self.pos + len >= self.bytes.len() {
                        break;
                    }
                    self.pos += len;
                }
                b'\n' | b'\r' if newlines => self.pos += 1,
                _ => break,
            }
        }
    }

    /// Blank and comment-only lines may precede the expression, whose own line must not be
    /// indented (CPython raises `IndentationError`).
    fn skip_leading_lines(&mut self) -> Result<(), Error> {
        loop {
            let line_start = self.pos;
            self.skip_trivia(false);
            match self.peek() {
                Some(b'\n' | b'\r') => self.pos += 1,
                Some(_) if line_start > 0 && self.pos > line_start => {
                    self.pos = line_start;
                    return Err(self.error());
                }
                _ => return Ok(()),
            }
        }
    }

    fn at_logical_line_end(&self) -> bool {
        matches!(self.peek(), None | Some(b'\n' | b'\r'))
    }

    /// The `eval` input: an expression, or a tuple without parentheses.
    fn top_level(&mut self) -> Result<Term, Error> {
        let first = self.expression()?;
        self.skip_trivia(false);
        if self.peek() != Some(b',') {
            return Ok(first);
        }
        let mut values = vec![first.value];
        while self.peek() == Some(b',') {
            self.pos += 1;
            self.skip_trivia(false);
            if self.at_logical_line_end() {
                break;
            }
            values.push(self.expression()?.value);
            self.skip_trivia(false);
        }
        Ok(Term::other(Value::Tuple(values)))
    }

    /// A sum of unary terms, checked as `ast.literal_eval` checks `BinOp`.
    fn expression(&mut self) -> Result<Term, Error> {
        let mut left = self.unary()?;
        loop {
            self.skip_trivia(self.brackets > 0);
            let subtract = match self.peek() {
                Some(b'+') => false,
                Some(b'-') => true,
                _ => return Ok(left),
            };
            let at = self.pos;
            self.pos += 1;
            let right = self.unary()?;
            left = complex_sum(left, subtract, right)
                .ok_or(Error::InvalidLiteral(self.offset + at))?;
        }
    }

    fn unary(&mut self) -> Result<Term, Error> {
        self.skip_trivia(self.brackets > 0);
        let negative = match self.peek() {
            Some(b'+') => false,
            Some(b'-') => true,
            _ => return self.primary(),
        };
        let at = self.pos;
        self.pos += 1;
        self.enter()?;
        let operand = self.unary()?;
        self.depth -= 1;
        if operand.kind != Kind::Constant || !operand.is_number() {
            return Err(Error::InvalidLiteral(self.offset + at));
        }
        let value = if !negative {
            operand.value
        } else {
            match operand.value {
                Value::Int(value) => Value::Int(-value),
                Value::Float(value) => Value::Float(-value),
                Value::Complex { re, im } => Value::Complex { re: -re, im: -im },
                _ => unreachable!("checked numeric above"),
            }
        };
        Ok(Term {
            value,
            kind: Kind::Signed,
        })
    }

    fn primary(&mut self) -> Result<Term, Error> {
        match self.peek() {
            Some(b'(') => self.parenthesized(),
            Some(b'[') => self.list(),
            Some(b'{') => self.braced(),
            Some(b'0'..=b'9') => self.number(),
            Some(b'.') if matches!(self.peek_at(1), Some(b'0'..=b'9')) => self.number(),
            Some(b'\'' | b'"') => self.strings(),
            Some(byte) if byte.is_ascii_alphabetic() || byte == b'_' => {
                if self.string_prefix_len().is_some() {
                    return self.strings();
                }
                self.name()
            }
            _ => Err(self.error()),
        }
    }

    fn open(&mut self) -> Result<(), Error> {
        self.enter()?;
        self.brackets += 1;
        self.pos += 1;
        Ok(())
    }

    fn close(&mut self, byte: u8) -> Result<(), Error> {
        self.skip_trivia(true);
        self.expect(byte)?;
        self.brackets -= 1;
        self.depth -= 1;
        Ok(())
    }

    /// Comma-separated expressions up to `close`, with an optional trailing comma.
    fn elements(&mut self, close: u8) -> Result<Vec<Value>, Error> {
        let mut values = Vec::new();
        loop {
            self.skip_trivia(true);
            if self.peek() == Some(close) {
                return Ok(values);
            }
            values.push(self.expression()?.value);
            self.skip_trivia(true);
            if self.peek() != Some(b',') {
                return Ok(values);
            }
            self.pos += 1;
        }
    }

    fn parenthesized(&mut self) -> Result<Term, Error> {
        self.open()?;
        self.skip_trivia(true);
        if self.peek() == Some(b')') {
            self.close(b')')?;
            return Ok(Term::other(Value::Tuple(Vec::new())));
        }
        let first = self.expression()?;
        self.skip_trivia(true);
        if self.peek() != Some(b',') {
            self.close(b')')?;
            return Ok(first);
        }
        self.pos += 1;
        let mut values = vec![first.value];
        values.extend(self.elements(b')')?);
        self.close(b')')?;
        Ok(Term::other(Value::Tuple(values)))
    }

    fn list(&mut self) -> Result<Term, Error> {
        self.open()?;
        let values = self.elements(b']')?;
        self.close(b']')?;
        Ok(Term::other(Value::List(values)))
    }

    fn braced(&mut self) -> Result<Term, Error> {
        self.open()?;
        self.skip_trivia(true);
        if self.peek() == Some(b'}') {
            self.close(b'}')?;
            return Ok(Term::other(Value::Dict(Vec::new())));
        }
        let first = self.expression()?.value;
        self.skip_trivia(true);
        if self.peek() != Some(b':') {
            let mut members = UniqueValues::default();
            members.insert(first, None)?;
            if self.peek() == Some(b',') {
                self.pos += 1;
                for member in self.elements(b'}')? {
                    members.insert(member, None)?;
                }
            }
            self.close(b'}')?;
            return Ok(Term::other(Value::Set(members.keys)));
        }
        let mut entries = UniqueValues::default();
        let mut key = first;
        loop {
            self.expect(b':')?;
            let value = self.expression()?.value;
            entries.insert(key, Some(value))?;
            self.skip_trivia(true);
            if self.peek() != Some(b',') {
                break;
            }
            self.pos += 1;
            self.skip_trivia(true);
            if self.peek() == Some(b'}') {
                break;
            }
            key = self.expression()?.value;
            self.skip_trivia(true);
        }
        self.close(b'}')?;
        Ok(Term::other(Value::Dict(entries.into_entries())))
    }

    fn identifier(&mut self) -> &[u8] {
        let start = self.pos;
        while matches!(self.peek(), Some(byte) if byte.is_ascii_alphanumeric() || byte == b'_') {
            self.pos += 1;
        }
        &self.bytes[start..self.pos]
    }

    fn name(&mut self) -> Result<Term, Error> {
        let at = self.pos;
        let value = match self.identifier() {
            b"True" => Value::Bool(true),
            b"False" => Value::Bool(false),
            b"None" => Value::None,
            b"set" => {
                self.skip_trivia(self.brackets > 0);
                self.expect(b'(')?;
                self.skip_trivia(true);
                self.expect(b')')?;
                return Ok(Term::other(Value::Set(Vec::new())));
            }
            _ => return Err(Error::InvalidLiteral(self.offset + at)),
        };
        Ok(Term {
            value,
            kind: Kind::Constant,
        })
    }

    /// Digits with single underscores between them, as CPython's `digitpart`.
    fn digits(&mut self, radix: u32, out: &mut String) -> Result<(), Error> {
        let start = out.len();
        loop {
            match self.peek() {
                Some(byte) if (byte as char).is_digit(radix) => {
                    out.push(byte as char);
                    self.pos += 1;
                }
                Some(b'_')
                    if out.len() > start
                        && matches!(self.peek_at(1), Some(next) if (next as char).is_digit(radix)) =>
                {
                    self.pos += 1;
                }
                _ => break,
            }
        }
        if out.len() == start {
            return Err(self.error());
        }
        Ok(())
    }

    fn number(&mut self) -> Result<Term, Error> {
        let start = self.pos;
        let radix = match (
            self.peek(),
            self.peek_at(1).map(|byte| byte.to_ascii_lowercase()),
        ) {
            (Some(b'0'), Some(b'x')) => Some(16),
            (Some(b'0'), Some(b'o')) => Some(8),
            (Some(b'0'), Some(b'b')) => Some(2),
            _ => None,
        };
        let mut text = String::new();
        let value = if let Some(radix) = radix {
            self.pos += 2;
            if self.peek() == Some(b'_') {
                self.pos += 1;
            }
            self.digits(radix, &mut text)?;
            Value::Int(BigInt::parse_bytes(text.as_bytes(), radix).ok_or(self.error())?)
        } else {
            let mut is_float = false;
            if self.peek() != Some(b'.') {
                self.digits(10, &mut text)?;
            }
            let integer_digits = text.clone();
            if self.peek() == Some(b'.') {
                is_float = true;
                self.pos += 1;
                text.push('.');
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.digits(10, &mut text)?;
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                is_float = true;
                self.pos += 1;
                text.push('e');
                if let Some(sign @ (b'+' | b'-')) = self.peek() {
                    text.push(sign as char);
                    self.pos += 1;
                }
                self.digits(10, &mut text)?;
            }
            if matches!(self.peek(), Some(b'j' | b'J')) {
                self.pos += 1;
                let im = text.parse::<f64>().map_err(|_| self.error())?;
                Value::Complex { re: 0.0, im }
            } else if is_float {
                Value::Float(text.parse::<f64>().map_err(|_| self.error())?)
            } else {
                if integer_digits.len() > 1
                    && integer_digits.starts_with('0')
                    && integer_digits.bytes().any(|digit| digit != b'0')
                {
                    return Err(Error::InvalidLiteral(self.offset + start));
                }
                Value::Int(BigInt::parse_bytes(integer_digits.as_bytes(), 10).ok_or(self.error())?)
            }
        };
        if matches!(self.peek(), Some(byte) if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.')
        {
            return Err(self.error());
        }
        Ok(Term {
            value,
            kind: Kind::Constant,
        })
    }

    /// The length of a valid string prefix (`r`, `u`, `b`, `br`, `rb`, any case) directly
    /// followed by a quote.
    fn string_prefix_len(&self) -> Option<usize> {
        let mut len = 0;
        while matches!(self.peek_at(len), Some(byte) if byte.is_ascii_alphabetic()) && len < 3 {
            len += 1;
        }
        if !matches!(self.peek_at(len), Some(b'\'' | b'"')) {
            return None;
        }
        let prefix: Vec<u8> = self.bytes[self.pos..self.pos + len]
            .iter()
            .map(u8::to_ascii_lowercase)
            .collect();
        matches!(prefix.as_slice(), b"" | b"r" | b"u" | b"b" | b"br" | b"rb").then_some(len)
    }

    /// Adjacent string literals concatenate; `str` and `bytes` cannot mix.
    fn strings(&mut self) -> Result<Term, Error> {
        let mut text: Option<String> = None;
        let mut bytes: Option<Vec<u8>> = None;
        loop {
            let at = self.pos;
            let Some(prefix_len) = self.string_prefix_len() else {
                break;
            };
            let prefix = &self.bytes[self.pos..self.pos + prefix_len];
            let raw = prefix.iter().any(|byte| byte.eq_ignore_ascii_case(&b'r'));
            let is_bytes = prefix.iter().any(|byte| byte.eq_ignore_ascii_case(&b'b'));
            self.pos += prefix_len;
            let mut out = Vec::new();
            self.string_body(raw, is_bytes, &mut out)?;
            if is_bytes {
                if text.is_some() {
                    return Err(Error::InvalidLiteral(self.offset + at));
                }
                bytes.get_or_insert_with(Vec::new).extend(out);
            } else {
                if bytes.is_some() {
                    return Err(Error::InvalidLiteral(self.offset + at));
                }
                let piece =
                    String::from_utf8(out).map_err(|_| Error::InvalidLiteral(self.offset + at))?;
                text.get_or_insert_with(String::new).push_str(&piece);
            }
            self.skip_trivia(self.brackets > 0);
        }
        let value = match (text, bytes) {
            (Some(text), None) => Value::Str(text),
            (None, Some(bytes)) => Value::Bytes(bytes),
            _ => return Err(self.error()),
        };
        Ok(Term {
            value,
            kind: Kind::Constant,
        })
    }

    /// One quoted body, decoded into UTF-8 (`str`) or raw bytes (`bytes`).
    fn string_body(&mut self, raw: bool, is_bytes: bool, out: &mut Vec<u8>) -> Result<(), Error> {
        let quote = self.bytes[self.pos];
        let triple = self.peek_at(1) == Some(quote) && self.peek_at(2) == Some(quote);
        self.pos += if triple { 3 } else { 1 };
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.error());
            };
            if byte == quote
                && (!triple || (self.peek_at(1) == Some(quote) && self.peek_at(2) == Some(quote)))
            {
                self.pos += if triple { 3 } else { 1 };
                return Ok(());
            }
            match byte {
                b'\n' | b'\r' if !triple => return Err(self.error()),
                b'\\' if raw => {
                    let Some(next) = self.peek_at(1) else {
                        return Err(self.error());
                    };
                    out.push(b'\\');
                    self.pos += 1;
                    if next == b'\n' || next == b'\r' || next == quote || next == b'\\' {
                        out.push(next);
                        self.pos += 1;
                    }
                }
                b'\\' => {
                    self.pos += 1;
                    self.escape(is_bytes, out)?;
                }
                byte if is_bytes && !byte.is_ascii() => return Err(self.error()),
                byte => {
                    out.push(byte);
                    self.pos += 1;
                }
            }
        }
    }

    fn escape(&mut self, is_bytes: bool, out: &mut Vec<u8>) -> Result<(), Error> {
        let Some(byte) = self.peek() else {
            return Err(self.error());
        };
        self.pos += 1;
        let simple = match byte {
            b'\n' => return Ok(()),
            b'\r' => {
                if self.peek() == Some(b'\n') {
                    self.pos += 1;
                }
                return Ok(());
            }
            b'\\' | b'\'' | b'"' => byte,
            b'a' => 0x07,
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'v' => 0x0b,
            b'0'..=b'7' => {
                let mut code = u32::from(byte - b'0');
                for _ in 0..2 {
                    match self.peek() {
                        Some(digit @ b'0'..=b'7') => {
                            code = code * 8 + u32::from(digit - b'0');
                            self.pos += 1;
                        }
                        _ => break,
                    }
                }
                // Bytes keep the low eight bits of `\400`-`\777`, as CPython does.
                return self.push_code(if is_bytes { code & 0xff } else { code }, is_bytes, out);
            }
            b'x' => {
                let code = self.hex(2)?;
                return self.push_code(code, is_bytes, out);
            }
            b'u' if !is_bytes => {
                let code = self.hex(4)?;
                return self.push_code(code, is_bytes, out);
            }
            b'U' if !is_bytes => {
                let code = self.hex(8)?;
                return self.push_code(code, is_bytes, out);
            }
            b'N' if !is_bytes => return Err(self.error()),
            _ => {
                // Unknown escapes keep the backslash (a `SyntaxWarning` in CPython).
                out.push(b'\\');
                self.pos -= 1;
                return Ok(());
            }
        };
        out.push(simple);
        Ok(())
    }

    fn hex(&mut self, count: usize) -> Result<u32, Error> {
        let mut code = 0u32;
        for _ in 0..count {
            let digit = self
                .peek()
                .and_then(|byte| (byte as char).to_digit(16))
                .ok_or(self.error())?;
            code = code * 16 + digit;
            self.pos += 1;
        }
        Ok(code)
    }

    fn push_code(&self, code: u32, is_bytes: bool, out: &mut Vec<u8>) -> Result<(), Error> {
        if is_bytes {
            out.push(u8::try_from(code).map_err(|_| self.error())?);
            return Ok(());
        }
        let ch = char::from_u32(code).ok_or(self.error())?;
        let mut buffer = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buffer).as_bytes());
        Ok(())
    }
}

/// `left + right` or `left - right` as `ast.literal_eval` permits: a signed real on the
/// left and an unsigned complex constant on the right, combined with CPython 3.14's
/// mixed-mode rules, which leave the imaginary part untouched by the real operand.
fn complex_sum(left: Term, subtract: bool, right: Term) -> Option<Term> {
    if left.kind == Kind::Other || right.kind != Kind::Constant {
        return None;
    }
    let real = match &left.value {
        Value::Int(value) => value.to_f64().filter(|value| value.is_finite())?,
        Value::Float(value) => *value,
        _ => return None,
    };
    let Value::Complex { re, im } = right.value else {
        return None;
    };
    let value = if subtract {
        Value::Complex {
            re: real - re,
            im: -im,
        }
    } else {
        Value::Complex { re: real + re, im }
    };
    Some(Term::other(value))
}

/// Python equality for hashable literal values: numbers compare by value across `bool`,
/// `int`, `float`, and `complex`, so `1`, `1.0`, `True`, and `(1+0j)` are one key.
#[derive(Hash, PartialEq, Eq)]
enum KeyId {
    None,
    Int(BigInt),
    Float(u64),
    Complex(u64, u64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<KeyId>),
}

fn float_key(value: f64) -> KeyId {
    if value.fract() == 0.0
        && let Some(integer) = BigInt::from_f64(value)
    {
        return KeyId::Int(integer);
    }
    KeyId::Float(value.to_bits())
}

fn key_id(value: &Value) -> Result<KeyId, Error> {
    Ok(match value {
        Value::None => KeyId::None,
        Value::Bool(value) => KeyId::Int(BigInt::from(u8::from(*value))),
        Value::Int(value) => KeyId::Int(value.clone()),
        Value::Float(value) => float_key(*value),
        Value::Complex { re, im } if *im == 0.0 => float_key(*re),
        Value::Complex { re, im } => KeyId::Complex((re + 0.0).to_bits(), (im + 0.0).to_bits()),
        Value::Str(text) => KeyId::Str(text.clone()),
        Value::Bytes(bytes) => KeyId::Bytes(bytes.clone()),
        Value::Tuple(values) => KeyId::Tuple(values.iter().map(key_id).collect::<Result<_, _>>()?),
        value @ (Value::List(_) | Value::Dict(_) | Value::Set(_)) => {
            return Err(Error::Unhashable(value.type_name()));
        }
    })
}

/// Dict entries or set members in first-seen order: a repeated key keeps its first
/// position and, for dicts, takes the latest value.
#[derive(Default)]
struct UniqueValues {
    keys: Vec<Value>,
    values: Vec<Value>,
    index: HashMap<KeyId, usize>,
}

impl UniqueValues {
    fn insert(&mut self, key: Value, value: Option<Value>) -> Result<(), Error> {
        match self.index.entry(key_id(&key)?) {
            Entry::Occupied(slot) => {
                if let Some(value) = value {
                    self.values[*slot.get()] = value;
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(self.keys.len());
                self.keys.push(key);
                self.values.extend(value);
            }
        }
        Ok(())
    }

    fn into_entries(self) -> Vec<(Value, Value)> {
        self.keys.into_iter().zip(self.values).collect()
    }
}
