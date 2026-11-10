export const codeSamples = {
  agents: {
    identityScope: "Agent.Invoke",
    killSwitchAuthHeader: "X-API-Key",
    killSwitchPayloadJson: '{"reason": "manual kill switch"}',
  },
  guardrails: {
    testPanelParamsJson: '{"forbidden_topics": ["tax", "finance"]}',
    teamGuardrailParamsJson: '{"forward_api_key": true, "headers": {"X-Custom": "value"}}',
    teamGuardrailInfoJson: '{"description": "Detects PII in requests"}',
    customCodeHook: "(pre_call)",
    customCodeInputJson: '{"texts": ["test message"], ...}',
    gardenIdPrefix: "litellm/",
    toolParamPath: "messages[0].content",
    toolParamPattern: "^email@.*$",
    toolTypePattern: "^function$",
    evaluationResponseFormat: "response_format: json_schema",
  },
  mcpServers: {
    exampleServerName: "Zapier Gmail",
    idJagSigningAlgorithm: "RS256",
    oauthExtraParamsJson: '{\n  "organization": "my-org",\n  "team.id": "123"\n}',
    entraOboScope: "api://<app-id>/.default",
    upstreamTokenHeader: "Authorization",
    envJson: '{\n  "KEY": "value"\n}',
    toolSearchToolNames: "my_server-get_rates\nmy_server-list_accounts",
  },
  memory: {
    metadataJson: '{"tags": ["example"]}',
  },
  vectorStores: {
    metadataJson: '{"key": "value"}',
  },
  playground: {
    responsesSessionCurl: (sessionId: string) => `curl -X POST "your-proxy-url/v1/responses" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "your-model",
    "input": [{"role": "user", "content": "your message", "type": "message"}],
    "previous_response_id": "${sessionId}",
    "stream": true
  }'`,
  },
  routerSettings: {
    baggageKey: "service",
    tagPrefix: "app:",
    anyValuePreviewSeparator: "=x → ",
    anyValuePreviewValue: "x",
  },
  models: {
    perModelLimitsJson: '{"gpt-4": 100, "claude-v1": 200}',
    perModelMaxTokensJson: '{"gpt-4": 4096}',
    litellmParamsJson: '{\n  "drop_params": true\n}',
    advancedLitellmParamsJson:
      '{\n                  "rpm": 100,\n                  "timeout": 0,\n                  "stream_timeout": 0\n                }',
    advancedModelInfoJson: '{\n                  "mode": "chat"\n                }',
    openingPromptExample: '- "what is the capital of France?" -> SIMPLE',
    toolSemanticsJson: '{"observe": ["read_file"], "mutate": ["edit_file"]}',
    reminderMarkerOpen: "<note>",
    reminderMarkerClose: "</note>",
    noAliasesYamlComment: "  # No aliases configured yet",
    noModelGroupAliasesYamlComment: "    # No aliases configured yet",
  },
  teams: {
    vaultConfigJson: '{"namespace": "admin", "mount": "secret", "path_prefix": "litellm"}',
    memberEmail: "user@example.com",
    memberEmails: "example1@test.com, example2@test.com",
    callbackEnvVar: (paramName: string) => `os.environ/${paramName.toUpperCase()}`,
  },
  passThrough: {
    headersJson: '{"Authorization": "Bearer your-token", "Content-Type": "application/json"}',
    targetPrefix: "bria",
    guardrailRequestQuery: "• query",
    guardrailRequestDocuments: "• documents[*].text",
    guardrailRequestMessages: "• messages[*].content",
    guardrailAddQuery: "+ query",
    guardrailAddDocuments: "+ documents[*]",
    guardrailResponseResults: "• results[*].text",
    guardrailResponseChoices: "• choices[*].message.content",
    guardrailAddResults: "+ results[*]",
  },
  routingGroups: {
    cacheParamsJson: '{ "ttl": 3600 }',
  },
  users: {
    bulkTemplateTeamsColumn: "teams",
    bulkTemplateModelsColumn: "models",
    ssoGroupsClaim: "groups",
  },
  logging: {
    premiumLangfuse: "✨ langfuse-logging",
    premiumDatadog: "✨ datadog-logging",
  },
} as const;
