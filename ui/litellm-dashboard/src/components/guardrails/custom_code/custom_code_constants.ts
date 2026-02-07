// Custom Code Guardrail Constants

export const DEFAULT_CUSTOM_CODE = `def apply_guardrail(inputs, request_data, input_type):
    # inputs: contains texts, images, tools, tool_calls, structured_messages, model
    # request_data: contains model, user_id, team_id, end_user_id, metadata
    # input_type: "request" or "response"
    
    for text in inputs["texts"]:
        # Example: Block if SSN pattern is detected
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected in message")
    
    return allow()
`;

export const CUSTOM_CODE_PRIMITIVES = {
  "Regex Functions": [
    {
      name: "regex_match",
      signature: "regex_match(text, pattern)",
      description: "Returns True if pattern found in text",
    },
    {
      name: "regex_replace",
      signature: "regex_replace(text, pattern, replacement)",
      description: "Replace all matches of pattern with replacement",
    },
    {
      name: "regex_find_all",
      signature: "regex_find_all(text, pattern)",
      description: "Return list of all matches",
    },
  ],
  "JSON Functions": [
    {
      name: "json_parse",
      signature: "json_parse(text)",
      description: "Parse JSON string, returns None on error",
    },
    {
      name: "json_stringify",
      signature: "json_stringify(obj)",
      description: "Convert object to JSON string",
    },
    {
      name: "json_schema_valid",
      signature: "json_schema_valid(obj, schema)",
      description: "Validate object against JSON schema",
    },
  ],
  "URL Functions": [
    {
      name: "extract_urls",
      signature: "extract_urls(text)",
      description: "Extract all URLs from text",
    },
    {
      name: "is_valid_url",
      signature: "is_valid_url(url)",
      description: "Check if URL is valid",
    },
    {
      name: "all_urls_valid",
      signature: "all_urls_valid(text)",
      description: "Check all URLs in text are valid",
    },
  ],
  "Code Detection": [
    {
      name: "detect_code",
      signature: "detect_code(text)",
      description: "Returns True if code detected",
    },
    {
      name: "detect_code_languages",
      signature: "detect_code_languages(text)",
      description: "Returns list of detected languages",
    },
    {
      name: "contains_code_language",
      signature: 'contains_code_language(text, ["sql", "python"])',
      description: "Check for specific languages",
    },
  ],
  "Text Utilities": [
    {
      name: "contains",
      signature: "contains(text, substring)",
      description: "Check if substring exists in text",
    },
    {
      name: "contains_any",
      signature: "contains_any(text, [substr1, substr2])",
      description: "Check if any substring exists",
    },
    {
      name: "word_count",
      signature: "word_count(text)",
      description: "Count words in text",
    },
    {
      name: "char_count",
      signature: "char_count(text)",
      description: "Count characters in text",
    },
    {
      name: "lower",
      signature: "lower(text)",
      description: "Convert text to lowercase",
    },
    {
      name: "upper",
      signature: "upper(text)",
      description: "Convert text to uppercase",
    },
    {
      name: "trim",
      signature: "trim(text)",
      description: "Remove leading/trailing whitespace",
    },
  ],
};

export const CUSTOM_CODE_EXAMPLES = {
  blockSSN: `def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected")
    return allow()
`,

  redactEmail: `def apply_guardrail(inputs, request_data, input_type):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    modified = []
    for text in inputs["texts"]:
        modified.append(regex_replace(text, pattern, "[EMAIL REDACTED]"))
    return modify(texts=modified)
`,

  blockSQL: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "request":
        return allow()
    for text in inputs["texts"]:
        if contains_code_language(text, ["sql"]):
            return block("SQL code not allowed")
    return allow()
`,

  validateJSON: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "response":
        return allow()
    
    schema = {
        "type": "object",
        "required": ["name", "value"]
    }
    
    for text in inputs["texts"]:
        obj = json_parse(text)
        if obj is None:
            return block("Invalid JSON response")
        if not json_schema_valid(obj, schema):
            return block("Response missing required fields")
    return allow()
`,

  checkURLs: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "response":
        return allow()
    for text in inputs["texts"]:
        if not all_urls_valid(text):
            return block("Response contains invalid URLs")
    return allow()
`,

  combined: `def apply_guardrail(inputs, request_data, input_type):
    modified = []
    
    for text in inputs["texts"]:
        # Redact SSN
        text = regex_replace(text, r"\\d{3}-\\d{2}-\\d{4}", "[SSN]")
        # Redact credit cards
        text = regex_replace(text, r"\\d{16}", "[CARD]")
        modified.append(text)
    
    # Block SQL in requests
    if input_type == "request":
        for text in inputs["texts"]:
            if contains_code_language(text, ["sql"]):
                return block("SQL injection blocked")
    
    return modify(texts=modified)
`,
};
