use std::fmt::Write;

use serde_json::{Map, Value};

pub(crate) fn format_function_definitions(tools: &[Map<String, Value>], out: &mut String) {
    out.push_str("namespace functions {\n\n");

    for tool in tools {
        let Some(function) = extract_function_dict(tool) else {
            continue;
        };

        let Some(name) = function.get("name").and_then(|v| v.as_str()) else {
            continue;
        };

        if let Some(desc) = function.get("description").and_then(|v| v.as_str())
            && !desc.is_empty()
        {
            let _ = writeln!(out, "// {desc}");
        }

        let parameters = function
            .get("parameters")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let has_properties = parameters
            .get("properties")
            .and_then(|v| v.as_object())
            .is_some_and(|p| !p.is_empty());

        if has_properties {
            let _ = writeln!(out, "type {name} = (_: {{");
            format_object_parameters(&parameters, 0, out);
            out.push_str("}) => any;\n");
        } else {
            let _ = writeln!(out, "type {name} = () => any;");
        }

        out.push('\n');
    }

    out.push_str("} // namespace functions");
}

fn extract_function_dict(tool: &Map<String, Value>) -> Option<Map<String, Value>> {
    if let Some(function) = tool.get("function").and_then(|v| v.as_object()) {
        return Some(function.clone());
    }

    let params = tool
        .get("input_schema")
        .or_else(|| tool.get("parameters"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    Some(Map::from_iter([
        (
            "name".to_string(),
            tool.get("name").cloned().unwrap_or(Value::Null),
        ),
        (
            "description".to_string(),
            tool.get("description").cloned().unwrap_or(Value::Null),
        ),
        ("parameters".to_string(), Value::Object(params)),
    ]))
}

fn format_object_parameters(parameters: &Map<String, Value>, indent: usize, out: &mut String) {
    let Some(properties) = parameters.get("properties").and_then(|v| v.as_object()) else {
        return;
    };

    let required: Vec<&str> = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let prefix = " ".repeat(indent);

    for (key, props) in properties {
        let props_obj = props.as_object();

        if let Some(obj) = props_obj
            && let Some(desc) = obj.get("description").and_then(|v| v.as_str())
            && !desc.is_empty()
        {
            let _ = writeln!(out, "{prefix}// {desc}");
        }

        let question = if required.contains(&key.as_str()) {
            ""
        } else {
            "?"
        };

        let mut type_buf = String::new();
        if let Some(p) = props_obj {
            format_type(p, indent, &mut type_buf);
        } else {
            type_buf.push_str("any");
        }

        let _ = writeln!(out, "{prefix}{key}{question}: {type_buf},");
    }
}

fn format_type(props: &Map<String, Value>, indent: usize, out: &mut String) {
    let Some(type_val) = props.get("type").and_then(|v| v.as_str()) else {
        out.push_str("any");
        return;
    };

    match type_val {
        "string" => {
            if let Some(enum_vals) = props.get("enum").and_then(|v| v.as_array()) {
                let mut first = true;
                for v in enum_vals {
                    if !first {
                        out.push_str(" | ");
                    }
                    let _ = write!(out, "\"{}\"", v.as_str().unwrap_or(""));
                    first = false;
                }
            } else {
                out.push_str("string");
            }
        }
        "array" => {
            if let Some(items) = props.get("items").and_then(|v| v.as_object()) {
                format_type(items, indent, out);
            } else {
                out.push_str("any");
            }
            out.push_str("[]");
        }
        "object" => {
            out.push_str("{\n");
            format_object_parameters(props, indent + 2, out);
            out.push('}');
        }
        "integer" | "number" => {
            if let Some(enum_vals) = props.get("enum").and_then(|v| v.as_array()) {
                let mut first = true;
                for v in enum_vals {
                    if !first {
                        out.push_str(" | ");
                    }
                    if let Some(n) = v.as_i64() {
                        let _ = write!(out, "\"{n}\"");
                    } else if let Some(n) = v.as_f64() {
                        let _ = write!(out, "\"{n}\"");
                    } else {
                        let _ = write!(out, "\"{}\"", v.as_str().unwrap_or(""));
                    }
                    first = false;
                }
            } else {
                out.push_str("number");
            }
        }
        "boolean" => out.push_str("boolean"),
        "null" => out.push_str("null"),
        _ => out.push_str("any"),
    }
}
