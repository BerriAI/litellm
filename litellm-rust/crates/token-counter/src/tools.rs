//! Renders tool definitions the way `litellm.token_counter` does before
//! tokenizing them (the TypeScript-like namespace OpenAI appears to use).

use std::fmt::Write;

use super::Error;
use super::types::{EnumValue, Schema, SchemaType, ToolDefinition};

pub(super) fn format_function_definitions(tools: &[ToolDefinition]) -> Result<String, Error> {
    ToolFormatter::new(tools.len()).format(tools)
}

struct ToolFormatter {
    output: String,
}

impl ToolFormatter {
    fn new(tool_count: usize) -> Self {
        Self {
            output: String::with_capacity(tool_count.saturating_mul(128).saturating_add(48)),
        }
    }

    fn format(mut self, tools: &[ToolDefinition]) -> Result<String, Error> {
        self.output.push_str("namespace functions {\n\n");

        for tool in tools {
            self.write_function(tool)?;
        }

        self.output.push_str("} // namespace functions");
        Ok(self.output)
    }

    fn write_function(&mut self, tool: &ToolDefinition) -> Result<(), Error> {
        let (name, description, parameters) = resolve_function(tool);
        let Some(name) = name.filter(|name| !name.is_empty()) else {
            return Ok(());
        };

        if let Some(description) = description.filter(|description| !description.is_empty()) {
            self.output.push_str("// ");
            self.output.push_str(description);
            self.output.push('\n');
        }

        match parameters.filter(|parameters| {
            parameters
                .properties
                .as_ref()
                .is_some_and(|properties| !properties.is_empty())
        }) {
            Some(parameters) => {
                self.output.push_str("type ");
                self.output.push_str(name);
                self.output.push_str(" = (_: {\n");
                self.write_object_parameters(parameters, 0)?;
                self.output.push_str("\n}) => any;\n\n");
            }
            _ => {
                self.output.push_str("type ");
                self.output.push_str(name);
                self.output.push_str(" = () => any;\n\n");
            }
        }

        Ok(())
    }

    fn write_object_parameters(&mut self, parameters: &Schema, indent: usize) -> Result<(), Error> {
        let Some(properties) = parameters
            .properties
            .as_ref()
            .filter(|properties| !properties.is_empty())
        else {
            return Ok(());
        };
        let required = parameters.required.as_deref().unwrap_or_default();
        for (index, (key, props)) in properties.iter().enumerate() {
            if index > 0 {
                self.output.push('\n');
            }
            if let Some(description) = props
                .description
                .as_deref()
                .filter(|description| !description.is_empty())
            {
                self.write_indent(indent);
                self.output.push_str("// ");
                self.output.push_str(description);
                self.output.push('\n');
            }

            self.write_indent(indent);
            self.output.push_str(key);
            if !required.iter().any(|required| required == key) {
                self.output.push('?');
            }
            self.output.push_str(": ");
            self.write_type(props, indent)?;
            self.output.push(',');
        }

        Ok(())
    }

    fn write_type(&mut self, props: &Schema, indent: usize) -> Result<(), Error> {
        let Some(SchemaType::Name(schema_type)) = &props.schema_type else {
            self.output.push_str("any");
            return Ok(());
        };

        match schema_type.as_str() {
            "string" | "integer" | "number" => match &props.enum_values {
                Some(values) => self.write_enum(values),
                None if schema_type == "string" => self.output.push_str("string"),
                None => self.output.push_str("number"),
            },
            "array" => {
                let items = props.items.as_deref().ok_or(Error::ArrayItems)?;
                self.write_type(items, indent)?;
                self.output.push_str("[]");
            }
            "object" => {
                self.output.push_str("{\n");
                self.write_object_parameters(props, indent + 2)?;
                self.output.push_str("\n}");
            }
            "boolean" => self.output.push_str("boolean"),
            "null" => self.output.push_str("null"),
            _ => self.output.push_str("any"),
        }

        Ok(())
    }

    fn write_enum(&mut self, values: &[EnumValue]) {
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                self.output.push_str(" | ");
            }
            self.output.push('"');
            match value {
                EnumValue::Text(text) => self.output.push_str(text),
                EnumValue::Integer(number) => {
                    write!(self.output, "{number}").expect("writing to a String cannot fail");
                }
            }
            self.output.push('"');
        }
    }

    fn write_indent(&mut self, indent: usize) {
        for _ in 0..indent {
            self.output.push(' ');
        }
    }
}

fn resolve_function(tool: &ToolDefinition) -> (Option<&str>, Option<&str>, Option<&Schema>) {
    match &tool.function {
        Some(function) => (
            function.name.as_deref(),
            function.description.as_deref(),
            function.parameters.as_ref(),
        ),
        None => (
            tool.name.as_deref(),
            tool.description.as_deref(),
            tool.input_schema.as_ref().or(tool.parameters.as_ref()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_tools(json: &str) -> Vec<ToolDefinition> {
        serde_json::from_str(json).expect("tool fixture parses")
    }

    #[test]
    fn empty_tool_list_renders_an_empty_namespace() {
        assert_eq!(
            format_function_definitions(&[]).expect("empty tool list renders"),
            "namespace functions {\n\n} // namespace functions"
        );
    }

    #[test]
    fn unnamed_tools_are_skipped_and_function_shape_takes_precedence() {
        let tools = parse_tools(
            r#"[
                {},
                {"name":""},
                {"name":"ignored","function":{"description":"missing a function name"}},
                {"name":"ping","description":""}
            ]"#,
        );

        assert_eq!(
            format_function_definitions(&tools).expect("tools render"),
            "namespace functions {\n\ntype ping = () => any;\n\n} // namespace functions"
        );
    }

    #[test]
    fn empty_or_missing_properties_render_a_no_argument_function() {
        let tools = parse_tools(
            r#"[
                {"name":"missing","input_schema":{"type":"object"}},
                {"name":"empty","input_schema":{"type":"object","properties":{}}}
            ]"#,
        );

        assert_eq!(
            format_function_definitions(&tools).expect("tools render"),
            concat!(
                "namespace functions {\n\n",
                "type missing = () => any;\n\n",
                "type empty = () => any;\n\n",
                "} // namespace functions"
            )
        );
    }

    #[test]
    fn anthropic_parameters_render_all_supported_types() {
        let tools = parse_tools(
            r#"[{
                "name":"inspect",
                "description":"Inspect a value",
                "input_schema":{
                    "type":"object",
                    "properties":{
                        "text":{"type":"string"},
                        "count":{"type":"integer","description":"Number of attempts"},
                        "ratio":{"type":"number"},
                        "enabled":{"type":"boolean"},
                        "nothing":{"type":"null"},
                        "unknown":{"type":"custom"},
                        "union":{"type":["string","null"]},
                        "labels":{"type":"array","items":{"type":"string"}},
                        "config":{"type":"object","properties":{"retries":{"type":"integer"}},"required":["retries"]},
                        "mode":{"type":"string","enum":["fast",2]}
                    },
                    "required":["text"]
                }
            }]"#,
        );

        assert_eq!(
            format_function_definitions(&tools).expect("tool renders"),
            concat!(
                "namespace functions {\n\n",
                "// Inspect a value\n",
                "type inspect = (_: {\n",
                "text: string,\n",
                "// Number of attempts\n",
                "count?: number,\n",
                "ratio?: number,\n",
                "enabled?: boolean,\n",
                "nothing?: null,\n",
                "unknown?: any,\n",
                "union?: any,\n",
                "labels?: string[],\n",
                "config?: {\n",
                "  retries: number,\n",
                "},\n",
                "mode?: \"fast\" | \"2\",\n",
                "}) => any;\n\n",
                "} // namespace functions"
            )
        );
    }

    #[test]
    fn input_schema_takes_precedence_over_legacy_parameters() {
        let tools = parse_tools(
            r#"[{
                "name":"choose",
                "input_schema":{"type":"object","properties":{"current":{"type":"string"}}},
                "parameters":{"type":"object","properties":{"legacy":{"type":"string"}}}
            }]"#,
        );

        let rendered = format_function_definitions(&tools).expect("tool renders");
        assert!(rendered.contains("current?: string,"));
        assert!(!rendered.contains("legacy"));
    }

    #[test]
    fn array_without_items_returns_an_error() {
        let tools = parse_tools(
            r#"[{"name":"broken","parameters":{"type":"object","properties":{"values":{"type":"array"}}}}]"#,
        );

        assert!(matches!(
            format_function_definitions(&tools),
            Err(Error::ArrayItems)
        ));
    }
}
