import type { ParseKeys } from "i18next";

export const CODE_TEMPLATES = {
  empty: {
    nameKey: "guardrails.customCode.templates.empty",
    code: `async def apply_guardrail(inputs, request_data, input_type):
    # inputs: {texts, images, tools, tool_calls, structured_messages, model}
    # request_data: {model, user_id, team_id, end_user_id, metadata}
    # input_type: "request" or "response"
    return allow()`,
  },
  blockSSN: {
    nameKey: "guardrails.customCode.templates.blockSsn",
    code: `def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected")
    return allow()`,
  },
  redactEmail: {
    nameKey: "guardrails.customCode.templates.redactEmail",
    code: `def apply_guardrail(inputs, request_data, input_type):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    modified = []
    for text in inputs["texts"]:
        modified.append(regex_replace(text, pattern, "[EMAIL REDACTED]"))
    return modify(texts=modified)`,
  },
  blockSQL: {
    nameKey: "guardrails.customCode.templates.blockSql",
    code: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "request":
        return allow()
    for text in inputs["texts"]:
        if contains_code_language(text, ["sql"]):
            return block("SQL code not allowed")
    return allow()`,
  },
  validateJSON: {
    nameKey: "guardrails.customCode.templates.validateJson",
    code: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "response":
        return allow()
    
    schema = {"type": "object", "required": ["name", "value"]}
    
    for text in inputs["texts"]:
        obj = json_parse(text)
        if obj is None:
            return block("Invalid JSON response")
        if not json_schema_valid(obj, schema):
            return block("Response missing required fields")
    return allow()`,
  },
  externalAPI: {
    nameKey: "guardrails.customCode.templates.externalApi",
    code: `async def apply_guardrail(inputs, request_data, input_type):
    # Call an external moderation API (async for non-blocking)
    for text in inputs["texts"]:
        response = await http_post(
            "https://api.example.com/moderate",
            body={"text": text, "user_id": request_data["user_id"]},
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            timeout=10
        )
        
        if not response["success"]:
            # API call failed, allow by default or block
            return allow()
        
        if response["body"].get("flagged"):
            return block(response["body"].get("reason", "Content flagged"))
    
    return allow()`,
  },
} satisfies Record<string, { nameKey: ParseKeys; code: string }>;

interface Primitive {
  name: string;
  descKey: ParseKeys;
}

interface PrimitiveCategory {
  id: string;
  labelKey: ParseKeys;
  primitives: Primitive[];
}

export const DEFAULT_OPEN_PRIMITIVE_CATEGORY = "returnValues";

export const PRIMITIVE_CATEGORIES: PrimitiveCategory[] = [
  {
    id: "returnValues",
    labelKey: "guardrails.customCode.primitives.categories.returnValues",
    primitives: [
      { name: "allow()", descKey: "guardrails.customCode.primitives.descriptions.allow" },
      { name: "block(reason)", descKey: "guardrails.customCode.primitives.descriptions.block" },
      { name: "flag(reason, metadata={})", descKey: "guardrails.customCode.primitives.descriptions.flag" },
      {
        name: "modify(texts=[], images=[], tool_calls=[])",
        descKey: "guardrails.customCode.primitives.descriptions.modify",
      },
    ],
  },
  {
    id: "httpRequests",
    labelKey: "guardrails.customCode.primitives.categories.httpRequests",
    primitives: [
      {
        name: "await http_request(url, method, headers, body)",
        descKey: "guardrails.customCode.primitives.descriptions.httpRequest",
      },
      { name: "await http_get(url, headers)", descKey: "guardrails.customCode.primitives.descriptions.httpGet" },
      {
        name: "await http_post(url, body, headers)",
        descKey: "guardrails.customCode.primitives.descriptions.httpPost",
      },
    ],
  },
  {
    id: "regex",
    labelKey: "guardrails.customCode.primitives.categories.regex",
    primitives: [
      { name: "regex_match(text, pattern)", descKey: "guardrails.customCode.primitives.descriptions.regexMatch" },
      {
        name: "regex_replace(text, pattern, replacement)",
        descKey: "guardrails.customCode.primitives.descriptions.regexReplace",
      },
      { name: "regex_find_all(text, pattern)", descKey: "guardrails.customCode.primitives.descriptions.regexFindAll" },
    ],
  },
  {
    id: "json",
    labelKey: "guardrails.customCode.primitives.categories.json",
    primitives: [
      { name: "json_parse(text)", descKey: "guardrails.customCode.primitives.descriptions.jsonParse" },
      { name: "json_stringify(obj)", descKey: "guardrails.customCode.primitives.descriptions.jsonStringify" },
      {
        name: "json_schema_valid(obj, schema)",
        descKey: "guardrails.customCode.primitives.descriptions.jsonSchemaValid",
      },
    ],
  },
  {
    id: "url",
    labelKey: "guardrails.customCode.primitives.categories.url",
    primitives: [
      { name: "extract_urls(text)", descKey: "guardrails.customCode.primitives.descriptions.extractUrls" },
      { name: "is_valid_url(url)", descKey: "guardrails.customCode.primitives.descriptions.isValidUrl" },
      { name: "all_urls_valid(text)", descKey: "guardrails.customCode.primitives.descriptions.allUrlsValid" },
    ],
  },
  {
    id: "codeDetection",
    labelKey: "guardrails.customCode.primitives.categories.codeDetection",
    primitives: [
      { name: "detect_code(text)", descKey: "guardrails.customCode.primitives.descriptions.detectCode" },
      {
        name: "detect_code_languages(text)",
        descKey: "guardrails.customCode.primitives.descriptions.detectCodeLanguages",
      },
      {
        name: 'contains_code_language(text, ["sql"])',
        descKey: "guardrails.customCode.primitives.descriptions.containsCodeLanguage",
      },
    ],
  },
  {
    id: "textUtilities",
    labelKey: "guardrails.customCode.primitives.categories.textUtilities",
    primitives: [
      { name: "contains(text, substring)", descKey: "guardrails.customCode.primitives.descriptions.contains" },
      {
        name: "contains_any(text, [substr1, substr2])",
        descKey: "guardrails.customCode.primitives.descriptions.containsAny",
      },
      { name: "word_count(text)", descKey: "guardrails.customCode.primitives.descriptions.wordCount" },
      { name: "char_count(text)", descKey: "guardrails.customCode.primitives.descriptions.charCount" },
      {
        name: "lower(text) / upper(text) / trim(text)",
        descKey: "guardrails.customCode.primitives.descriptions.stringTransforms",
      },
    ],
  },
];

export const MODE_OPTIONS: { value: string; labelKey: ParseKeys }[] = [
  { value: "pre_call", labelKey: "guardrails.customCode.modes.preCall" },
  { value: "post_call", labelKey: "guardrails.customCode.modes.postCall" },
  { value: "during_call", labelKey: "guardrails.customCode.modes.duringCall" },
  { value: "logging_only", labelKey: "guardrails.customCode.modes.loggingOnly" },
  { value: "pre_mcp_call", labelKey: "guardrails.customCode.modes.preMcpCall" },
  { value: "post_mcp_call", labelKey: "guardrails.customCode.modes.postMcpCall" },
  { value: "during_mcp_call", labelKey: "guardrails.customCode.modes.duringMcpCall" },
];

export const TEMPLATE_ITEMS: { value: string; labelKey: ParseKeys }[] = Object.entries(CODE_TEMPLATES).map(
  ([key, template]) => ({ value: key, labelKey: template.nameKey }),
);

export type ModeOption = { value: string; label: string };
