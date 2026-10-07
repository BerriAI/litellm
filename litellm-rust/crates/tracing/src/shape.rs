use std::{collections::BTreeSet, fmt::Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub struct ShapeLimits {
    pub nodes: usize,
    pub depth: usize,
    pub paths: usize,
    pub bytes: usize,
}

impl Default for ShapeLimits {
    fn default() -> Self {
        Self {
            nodes: 4096,
            depth: 16,
            paths: 256,
            bytes: 16384,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PayloadShape {
    pub field_paths: Vec<String>,
    pub truncated: bool,
}

impl PayloadShape {
    pub fn merge(&mut self, other: Self, limits: ShapeLimits) {
        let paths: BTreeSet<_> = self
            .field_paths
            .iter()
            .cloned()
            .chain(other.field_paths)
            .collect();
        if self.truncated
            || other.truncated
            || paths.len() > limits.paths
            || paths.iter().map(String::len).sum::<usize>() > limits.bytes
        {
            self.field_paths.clear();
            self.truncated = true;
            return;
        }
        self.field_paths = paths.into_iter().collect();
    }

    pub fn extract(value: &Value, limits: ShapeLimits) -> Self {
        Self::extract_source(value, limits)
    }

    pub fn extract_source(value: &impl ShapeSource, limits: ShapeLimits) -> Self {
        let mut collector = Collector {
            limits,
            nodes: 0,
            bytes: 0,
            paths: BTreeSet::new(),
        };
        if !collector.walk(value, "$", 0, false) {
            return Self {
                field_paths: vec![],
                truncated: true,
            };
        }
        Self {
            field_paths: collector.paths.into_iter().collect(),
            truncated: false,
        }
    }
}

struct Collector {
    limits: ShapeLimits,
    nodes: usize,
    bytes: usize,
    paths: BTreeSet<String>,
}

pub trait ShapeSource {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool;
}

pub struct ShapeVisitor<'a> {
    collector: &'a mut Collector,
    path: &'a str,
    depth: usize,
    dynamic: bool,
}

impl ShapeVisitor<'_> {
    pub fn field(&mut self, key: &str, child: &impl ShapeSource) -> bool {
        if key.len() > 128 {
            return false;
        }
        let next = if self.dynamic {
            format!("{}[*]", self.path)
        } else {
            format!("{}[{}]", self.path, quoted_key(key))
        };
        if self.collector.paths.insert(next.clone()) {
            self.collector.bytes += next.len();
            if self.collector.paths.len() > self.collector.limits.paths
                || self.collector.bytes > self.collector.limits.bytes
            {
                return false;
            }
        }
        self.collector
            .walk(child, &next, self.depth + 1, dynamic_keys(key))
    }

    pub fn item(&mut self, child: &impl ShapeSource) -> bool {
        self.collector.walk(
            child,
            &format!("{}[*]", self.path),
            self.depth + 1,
            self.dynamic,
        )
    }
}

impl ShapeSource for Value {
    fn visit(&self, visitor: &mut ShapeVisitor<'_>) -> bool {
        match self {
            Value::Object(fields) => fields.iter().all(|(key, child)| visitor.field(key, child)),
            Value::Array(items) => items.iter().all(|child| visitor.item(child)),
            _ => true,
        }
    }
}

impl Collector {
    fn walk(&mut self, value: &impl ShapeSource, path: &str, depth: usize, dynamic: bool) -> bool {
        self.nodes += 1;
        if self.nodes > self.limits.nodes || depth > self.limits.depth {
            return false;
        }
        value.visit(&mut ShapeVisitor {
            collector: self,
            path,
            depth,
            dynamic,
        })
    }
}

fn dynamic_keys(key: &str) -> bool {
    matches!(
        key,
        "metadata" | "properties" | "$defs" | "definitions" | "headers"
    )
}

fn quoted_key(key: &str) -> String {
    let mut quoted = String::from("'");
    for character in key.chars() {
        match character {
            '\'' => quoted.push_str("\\'"),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            '\u{8}' => quoted.push_str("\\b"),
            '\u{c}' => quoted.push_str("\\f"),
            character if character.is_control() && u32::from(character) < 0x20 => {
                write!(quoted, "\\u{:04x}", u32::from(character)).expect("writing to String");
            }
            character => quoted.push(character),
        }
    }
    quoted.push('\'');
    quoted
}
