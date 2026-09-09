//! Renders tool definitions the way `litellm.token_counter` does before
//! tokenizing them (the TypeScript-like namespace OpenAI appears to use).

use super::TokenCountError;
use super::types::{EnumValue, FunctionDefinition, Schema, SchemaType, ToolDefinition};

pub(super) fn format_function_definitions(
    tools: &[ToolDefinition],
) -> Result<String, TokenCountError> {
    let mut lines = vec!["namespace functions {".to_string(), String::new()];
    for tool in tools {
        let function = resolve_function(tool);
        let Some(name) = function.name.as_deref().filter(|name| !name.is_empty()) else {
            continue;
        };
        if let Some(description) = function.description.as_deref().filter(|d| !d.is_empty()) {
            lines.push(format!("// {description}"));
        }
        let parameters = function.parameters.unwrap_or_default();
        match &parameters.properties {
            Some(properties) if !properties.is_empty() => {
                lines.push(format!("type {name} = (_: {{"));
                lines.push(format_object_parameters(&parameters, 0)?);
                lines.push("}) => any;".to_string());
            }
            _ => lines.push(format!("type {name} = () => any;")),
        }
        lines.push(String::new());
    }
    lines.push("} // namespace functions".to_string());
    Ok(lines.join("\n"))
}

fn resolve_function(tool: &ToolDefinition) -> FunctionDefinition {
    match &tool.function {
        Some(function) => function.clone(),
        None => FunctionDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool
                .input_schema
                .clone()
                .or_else(|| tool.parameters.clone()),
        },
    }
}

fn format_object_parameters(parameters: &Schema, indent: usize) -> Result<String, TokenCountError> {
    let Some(properties) = parameters.properties.as_ref().filter(|p| !p.is_empty()) else {
        return Ok(String::new());
    };
    let required = parameters.required.as_deref().unwrap_or_default();
    let mut lines = Vec::new();
    for (key, props) in properties {
        if let Some(description) = props.description.as_deref().filter(|d| !d.is_empty()) {
            lines.push(format!("// {description}"));
        }
        let question = if required.iter().any(|r| r == key) {
            ""
        } else {
            "?"
        };
        lines.push(format!("{key}{question}: {},", format_type(props, indent)?));
    }
    let pad = " ".repeat(indent);
    Ok(lines
        .iter()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n"))
}

fn format_type(props: &Schema, indent: usize) -> Result<String, TokenCountError> {
    let Some(SchemaType::Name(schema_type)) = &props.schema_type else {
        return Ok("any".to_string());
    };
    match schema_type.as_str() {
        "string" | "integer" | "number" => Ok(match &props.enum_values {
            Some(values) => format_enum(values),
            None if schema_type == "string" => "string".to_string(),
            None => "number".to_string(),
        }),
        "array" => {
            let items = props.items.as_deref().ok_or(TokenCountError::Unsupported(
                "array parameter without items".to_string(),
            ))?;
            Ok(format!("{}[]", format_type(items, indent)?))
        }
        "object" => Ok(format!(
            "{{\n{}\n}}",
            format_object_parameters(props, indent + 2)?
        )),
        "boolean" => Ok("boolean".to_string()),
        "null" => Ok("null".to_string()),
        _ => Ok("any".to_string()),
    }
}

fn format_enum(values: &[EnumValue]) -> String {
    values
        .iter()
        .map(|value| match value {
            EnumValue::Text(text) => format!("\"{text}\""),
            EnumValue::Integer(number) => format!("\"{number}\""),
        })
        .collect::<Vec<_>>()
        .join(" | ")
}
