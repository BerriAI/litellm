use std::collections::BTreeMap;

use schemars::Schema as JsonSchema;
use serde_json::{Value, json};
use utoipa::openapi::{
    Content, Info, OpenApi, OpenApiBuilder, Ref, RefOr, Required,
    path::{
        HttpMethod, OperationBuilder, Parameter, ParameterBuilder, ParameterIn, PathItem,
        PathsBuilder,
    },
    request_body::RequestBodyBuilder,
    response::{ResponseBuilder, ResponsesBuilder},
    schema::{ComponentsBuilder, Schema},
    security::{Http, HttpAuthScheme, SecurityScheme},
};

struct Endpoint {
    path: &'static str,
    method: HttpMethod,
    operation: &'static str,
    request: Option<&'static str>,
    response: &'static str,
    description: &'static str,
}

const ENDPOINTS: [Endpoint; 9] = [
    Endpoint {
        path: "/v1/traces",
        method: HttpMethod::Get,
        operation: "trace_list",
        request: Some("TraceListRequest"),
        response: "TracePage",
        description: "Search trace summaries. All clauses in q must match. Free text searches trace_id, name and input preview as case-insensitive substrings. key:value clauses match whole values, with * as a wildcard; double quotes group spaces or literal colons. Prefix keyed clauses with - to exclude matches. Unknown keys, missing values and malformed quoting return invalid_request. name describes the physical root; input searches its preview, falling back to the first nonempty agent or LLM preview; agent, model and attr.<key> match any span. root_status describes the root, has_error means any failed span. The default window is the last 24 hours. First-page ingestion timestamp cutoff is retained by the cursor. This excludes later-stamped exports, not delayed commits stamped before the cutoff, and is not a database transaction snapshot. Repeat q and sorting on continuation; omitted bounds reuse the cursor window. Sort ties use the canonical id in the same direction. The server may return fewer than page_size items to respect response limits. Continue until next_cursor is null. Span-derived metrics use the cutoff; spend enrichment is best effort",
    },
    Endpoint {
        path: "/v1/traces/histogram",
        method: HttpMethod::Get,
        operation: "trace_histogram",
        request: Some("TraceHistogramRequest"),
        response: "TraceHistogram",
        description: "Count matching traces in equal-width [start_ms,end_ms) buckets. Reuse the list window and as_of_ms for matching span-derived membership. failed counts traces with any failed span. Agent groups count successful traces under their alphabetically first agent name, or service when no agent name exists",
    },
    Endpoint {
        path: "/v1/traces/values/{field}",
        method: HttpMethod::Get,
        operation: "trace_values",
        request: Some("TraceValuesRequest"),
        response: "RunValues",
        description: "Return the most common distinct values among matching traces. contains is case-insensitive. limit is a top-K suggestion limit, not a pagination size. Reuse list window and as_of_ms for matching membership",
    },
    Endpoint {
        path: "/v1/traces/{id}",
        method: HttpMethod::Get,
        operation: "trace_get",
        request: Some("TraceNoQueryRequest"),
        response: "TraceMetadata",
        description: "Read summary and agent metadata by canonical id. trace_id is the original OTLP id, which can repeat across ownership scopes. Spans are read through the separate spans collection",
    },
    Endpoint {
        path: "/v1/traces/{id}/spans",
        method: HttpMethod::Get,
        operation: "trace_spans",
        request: Some("TraceSpanPageRequest"),
        response: "TraceSpansPage",
        description: "Read a bounded page of canonical spans. The cursor pins the graph version. Continue with the same page_size; a changed graph or expired reconstruction returns trace_changed and the traversal must restart",
    },
    Endpoint {
        path: "/v1/traces/{id}/spans/{span_id}",
        method: HttpMethod::Get,
        operation: "trace_span",
        request: Some("TraceNoQueryRequest"),
        response: "SpanDetail",
        description: "Read raw input, output and attributes for one span. UI rendering is performed by the client",
    },
    Endpoint {
        path: "/v1/traces/{id}/spans/{span_id}/error",
        method: HttpMethod::Get,
        operation: "trace_error",
        request: Some("TraceErrorPageRequest"),
        response: "SpanErrorPage",
        description: "Read bounded diagnostic text pages. Cursor validation detects content changes and requires restarting the traversal",
    },
    Endpoint {
        path: "/v1/traces/query",
        method: HttpMethod::Post,
        operation: "trace_query",
        request: Some("TraceQueryRequest"),
        response: "TraceSQLResponse",
        description: "Execute read-only ClickHouse SQL under authenticated row policies and fixed resource limits. Bind params with native {name:Type} placeholders. Results are always ClickHouse JSON; 64-bit integers may be strings. SQL callers control ORDER BY, LIMIT and keyset continuation. Exceeding a resource limit fails instead of returning partial success",
    },
    Endpoint {
        path: "/v1/traces/query/help",
        method: HttpMethod::Get,
        operation: "trace_query_help",
        request: Some("TraceNoQueryRequest"),
        response: "TraceQueryHelp",
        description: "Discover current SQL schema, logical views, scoped examples and resource limits",
    },
];

fn rewrite_refs(value: Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter_map(|(key, value)| {
                    if key == "$schema" || key == "$defs" {
                        return None;
                    }
                    let rewritten = match (key.as_str(), value) {
                        ("$ref", Value::String(reference)) => {
                            Value::String(reference.replace("#/$defs/", "#/components/schemas/"))
                        }
                        (_, value) => rewrite_refs(value),
                    };
                    Some((key, rewritten))
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(rewrite_refs).collect()),
        value => value,
    }
}

fn schema(value: Value) -> RefOr<Schema> {
    serde_json::from_value(rewrite_refs(value)).expect("Rust JSON Schema is an OpenAPI 3.1 schema")
}

fn query_parameters(name: &str, schemas: &BTreeMap<&str, JsonSchema>) -> Vec<Parameter> {
    let Some(properties) = schemas[name].get("properties").and_then(Value::as_object) else {
        return Vec::new();
    };
    properties
        .iter()
        .map(|(name, value)| {
            ParameterBuilder::new()
                .name(name)
                .parameter_in(ParameterIn::Query)
                .schema(Some(schema(value.clone())))
                .build()
        })
        .collect()
}

fn path_parameters(path: &str) -> impl Iterator<Item = Parameter> + '_ {
    path.split('/')
        .filter_map(|part| {
            part.strip_prefix('{')
                .and_then(|value| value.strip_suffix('}'))
        })
        .map(|name| {
            ParameterBuilder::new()
                .name(name)
                .parameter_in(ParameterIn::Path)
                .required(Required::True)
                .schema(Some(if name == "field" {
                    RefOr::Ref(Ref::from_schema_name("RunField"))
                } else {
                    schema(json!({"type":"string","minLength":1}))
                }))
                .build()
        })
}

fn endpoint(endpoint: &Endpoint, schemas: &BTreeMap<&str, JsonSchema>) -> PathItem {
    let success = ResponseBuilder::new()
        .description("Success")
        .content(
            "application/json",
            Content::new(Some(Ref::from_schema_name(endpoint.response))),
        )
        .build();
    let responses = [400, 401, 403, 404, 409, 413, 422, 500, 501, 503]
        .into_iter()
        .fold(
            ResponsesBuilder::new().response("200", success),
            |builder, status| {
                builder.response(
                    status.to_string(),
                    ResponseBuilder::new()
                        .description("Trace API problem")
                        .content(
                            "application/problem+json",
                            Content::new(Some(Ref::from_schema_name("TraceProblem"))),
                        )
                        .build(),
                )
            },
        )
        .build();
    let parameters = endpoint
        .request
        .filter(|_| endpoint.method == HttpMethod::Get)
        .map(|request| query_parameters(request, schemas))
        .unwrap_or_default();
    let request_body = endpoint
        .request
        .filter(|_| endpoint.method == HttpMethod::Post)
        .map(|request| {
            RequestBodyBuilder::new()
                .required(Some(Required::True))
                .content(
                    "application/json",
                    Content::new(Some(Ref::from_schema_name(request))),
                )
                .build()
        });
    let operation = OperationBuilder::new()
        .operation_id(Some(endpoint.operation))
        .security(utoipa::openapi::security::SecurityRequirement::new(
            "TraceBearer",
            [""; 0],
        ))
        .tag("agent tracing")
        .description(Some(endpoint.description))
        .parameters(Some(path_parameters(endpoint.path).chain(parameters)))
        .request_body(request_body)
        .responses(responses)
        .build();
    PathItem::new(endpoint.method.clone(), operation)
}

pub fn document(extra: BTreeMap<&'static str, JsonSchema>) -> OpenApi {
    let roots: BTreeMap<_, _> = crate::schema::schemas()
        .into_iter()
        .filter(|(name, _)| {
            *name == "RunField" || ENDPOINTS.iter().any(|endpoint| endpoint.response == *name)
        })
        .chain(crate::schema::api_schemas())
        .chain(extra)
        .collect();
    let definitions = roots
        .values()
        .flat_map(|root| {
            root.get("$defs")
                .and_then(Value::as_object)
                .into_iter()
                .flat_map(|defs| defs.iter())
        })
        .map(|(name, value)| (name.clone(), schema(value.clone())));
    let components = ComponentsBuilder::new()
        .schemas_from_iter(definitions)
        .schemas_from_iter(
            roots
                .iter()
                .map(|(name, root)| (*name, schema(root.as_value().clone()))),
        )
        .security_scheme(
            "TraceBearer",
            SecurityScheme::Http(Http::new(HttpAuthScheme::Bearer)),
        )
        .build();
    let ingest = OperationBuilder::new().operation_id(Some("trace_ingest"))
        .security(utoipa::openapi::security::SecurityRequirement::new("TraceBearer", [""; 0])).tag("agent tracing")
        .description(Some("Export OTLP traces as JSON or protobuf, optionally gzip compressed. The OTLP protocol defines payloads and responses: https://opentelemetry.io/docs/specs/otlp/. Ownership is derived from authentication, never payload attributes"))
        .request_body(Some(RequestBodyBuilder::new().required(Some(Required::True))
            .content("application/json", Content::new(Some(schema(json!({"type":"object"})))))
            .content("application/x-protobuf", Content::new(Some(schema(json!({"type":"string","format":"binary"})))))
            .build()))
        .responses([200,400,401,403,413,429,501,503].into_iter().fold(ResponsesBuilder::new(), |builder, status| {
            builder.response(status.to_string(), ResponseBuilder::new().description("OTLP protocol response")
                .content("application/json", Content::new(Some(schema(json!({"type":"object"})))))
                .content("application/x-protobuf", Content::new(Some(schema(json!({"type":"string","format":"binary"})))))
                .build())
        }).build()).build();
    let paths = ENDPOINTS
        .iter()
        .fold(PathsBuilder::new(), |builder, item| {
            builder.path(item.path, endpoint(item, &roots))
        })
        .path("/v1/traces", PathItem::new(HttpMethod::Post, ingest))
        .build();
    OpenApiBuilder::new()
        .info(Info::new("LiteLLM Trace API", "1"))
        .components(Some(components))
        .paths(paths)
        .security(Some([utoipa::openapi::security::SecurityRequirement::new(
            "TraceBearer",
            [""; 0],
        )]))
        .build()
}
