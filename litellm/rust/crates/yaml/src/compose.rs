//! PyYAML's `Composer` and the parser's choice of `implicit`: events become one node tree per
//! document, each node carrying the tag PyYAML gives it before construction.

use std::collections::HashMap;

use saphyr_parser::{Event, Parser, ScalarStyle, Span, StrInput};

use crate::{Error, Location, resolve};

/// A node's tag, spelled the way PyYAML prints it.
#[derive(Clone, Debug, PartialEq, Eq, strum::EnumString, strum::Display)]
pub enum Tag {
    #[strum(serialize = "tag:yaml.org,2002:null")]
    Null,
    #[strum(serialize = "tag:yaml.org,2002:bool")]
    Bool,
    #[strum(serialize = "tag:yaml.org,2002:int")]
    Int,
    #[strum(serialize = "tag:yaml.org,2002:float")]
    Float,
    #[strum(serialize = "tag:yaml.org,2002:str")]
    Str,
    #[strum(serialize = "tag:yaml.org,2002:merge")]
    Merge,
    #[strum(serialize = "tag:yaml.org,2002:seq")]
    Seq,
    #[strum(serialize = "tag:yaml.org,2002:map")]
    Map,
    #[strum(default)]
    Other(String),
}

#[derive(Clone, Debug)]
pub(crate) enum Kind {
    Scalar(String),
    Sequence(Vec<Node>),
    Mapping(Vec<(Node, Node)>),
}

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub(crate) tag: Tag,
    pub(crate) kind: Kind,
    pub(crate) at: Location,
    pub(crate) size: usize,
    /// How many levels the deepest descendant sits below this node.
    pub(crate) height: usize,
}

const MAX_DEPTH: usize = litellm_python_compat::MAX_DEPTH;
/// PyYAML shares an aliased node, while a `Value` tree copies it, so this bounds how far aliases
/// of aliases may multiply a small document.
const MAX_NODES: usize = 1_000_000;

/// The document's root node, `None` for an empty stream.
pub(crate) fn compose(text: &str) -> Result<Option<Node>, Error> {
    let mut composer = Composer {
        events: Parser::new_from_str(text),
        anchors: HashMap::new(),
        nodes: 0,
    };
    composer.next()?;
    let root = match composer.next()? {
        (Event::StreamEnd, _) => return Ok(None),
        _ => {
            let event = composer.next()?;
            composer.node(event, 0)?
        }
    };
    composer.next()?;
    match composer.next()? {
        (Event::StreamEnd, _) => Ok(Some(root)),
        (_, span) => Err(Error::MultipleDocuments { at: location(span) }),
    }
}

struct Composer<'input> {
    events: Parser<'input, StrInput<'input>>,
    anchors: HashMap<usize, Node>,
    nodes: usize,
}

impl<'input> Composer<'input> {
    fn next(&mut self) -> Result<(Event<'input>, Span), Error> {
        Ok(self
            .events
            .next()
            .transpose()?
            .expect("saphyr ends every stream with StreamEnd"))
    }

    fn node(&mut self, (event, span): (Event<'input>, Span), depth: usize) -> Result<Node, Error> {
        let at = location(span);
        if depth > MAX_DEPTH {
            return Err(Error::TooDeep {
                limit: MAX_DEPTH,
                at,
            });
        }
        let (anchor, node) = match event {
            Event::Alias(anchor) => return self.alias(anchor, at, depth),
            Event::Scalar(text, style, anchor, tag) => {
                let tag = scalar_tag(&text, style, tag.as_deref().map(full_tag));
                (anchor, leaf(tag, Kind::Scalar(text.into_owned()), at))
            }
            Event::SequenceStart(anchor, tag) => {
                let tag = collection_tag(tag.as_deref().map(full_tag), Tag::Seq);
                (anchor, self.sequence(tag, at, depth)?)
            }
            Event::MappingStart(anchor, tag) => {
                let tag = collection_tag(tag.as_deref().map(full_tag), Tag::Map);
                (anchor, self.mapping(tag, at, depth)?)
            }
            Event::Nothing
            | Event::StreamStart
            | Event::StreamEnd
            | Event::DocumentStart(_)
            | Event::DocumentEnd
            | Event::SequenceEnd
            | Event::MappingEnd => unreachable!("saphyr only starts a node with a node event"),
        };
        self.count(node.size)?;
        if anchor != 0 {
            self.anchors.insert(anchor, node.clone());
        }
        Ok(node)
    }

    fn alias(&mut self, anchor: usize, at: Location, depth: usize) -> Result<Node, Error> {
        let node = self
            .anchors
            .get(&anchor)
            .cloned()
            .ok_or(Error::RecursiveAlias { at })?;
        if depth + node.height > MAX_DEPTH {
            return Err(Error::TooDeep {
                limit: MAX_DEPTH,
                at,
            });
        }
        self.count(node.size)?;
        Ok(node)
    }

    fn count(&mut self, size: usize) -> Result<(), Error> {
        self.nodes += size;
        match self.nodes > MAX_NODES {
            true => Err(Error::TooLarge { limit: MAX_NODES }),
            false => Ok(()),
        }
    }

    fn sequence(&mut self, tag: Tag, at: Location, depth: usize) -> Result<Node, Error> {
        let mut items = Vec::new();
        loop {
            match self.next()? {
                (Event::SequenceEnd, _) => break,
                event => items.push(self.node(event, depth + 1)?),
            }
        }
        let size = 1 + items.iter().map(|item| item.size).sum::<usize>();
        let height = parent_height(items.iter());
        Ok(Node {
            tag,
            kind: Kind::Sequence(items),
            at,
            size,
            height,
        })
    }

    fn mapping(&mut self, tag: Tag, at: Location, depth: usize) -> Result<Node, Error> {
        let mut pairs = Vec::new();
        loop {
            match self.next()? {
                (Event::MappingEnd, _) => break,
                event => {
                    let key = self.node(event, depth + 1)?;
                    let value = self.next()?;
                    pairs.push((key, self.node(value, depth + 1)?));
                }
            }
        }
        let size = 1 + pairs
            .iter()
            .map(|(key, value)| key.size + value.size)
            .sum::<usize>();
        let height = parent_height(pairs.iter().flat_map(|(key, value)| [key, value]));
        Ok(Node {
            tag,
            kind: Kind::Mapping(pairs),
            at,
            size,
            height,
        })
    }
}

fn leaf(tag: Tag, kind: Kind, at: Location) -> Node {
    Node {
        tag,
        kind,
        at,
        size: 1,
        height: 0,
    }
}

fn parent_height<'node>(children: impl Iterator<Item = &'node Node>) -> usize {
    children.map(|child| child.height + 1).max().unwrap_or(0)
}

fn full_tag(tag: &saphyr_parser::Tag) -> String {
    format!("{}{}", tag.handle, tag.suffix)
}

/// A plain scalar without a tag, or any scalar tagged `!`, is resolved from its text, as PyYAML's
/// parser sets `implicit[0]`. Any other untagged scalar is a string.
fn scalar_tag(text: &str, style: ScalarStyle, tag: Option<String>) -> Tag {
    match tag.as_deref() {
        Some("!") => resolve::implicit(text),
        None if style == ScalarStyle::Plain => resolve::implicit(text),
        None => Tag::Str,
        Some(tag) => parse_tag(tag),
    }
}

fn collection_tag(tag: Option<String>, default: Tag) -> Tag {
    match tag.as_deref() {
        None | Some("!") => default,
        Some(tag) => parse_tag(tag),
    }
}

fn parse_tag(tag: &str) -> Tag {
    tag.parse().unwrap_or_else(|_| Tag::Other(tag.to_owned()))
}

fn location(span: Span) -> Location {
    Location {
        line: span.start.line(),
        column: span.start.col() + 1,
    }
}
