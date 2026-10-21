use askama::Template;

#[macro_rules_attribute::apply(crate::response_type)]
#[cfg_attr(feature = "schema", schemars(rename = "TraceQueryExample"))]
pub struct Example {
    pub name: String,
    pub sql: String,
}

pub struct Section<'a> {
    pub title: &'a str,
    pub body: &'a str,
}

#[derive(Template)]
#[template(path = "query_help.jinja", escape = "none")]
pub struct QueryGuide<'a> {
    pub sections: &'a [Section<'a>],
    pub examples: &'a [Example],
    pub gotchas: &'a [String],
}

impl QueryGuide<'_> {
    pub fn render(&self) -> Result<String, askama::Error> {
        Template::render(self)
    }
}
