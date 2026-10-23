import assert from "node:assert/strict";
import test from "node:test";
import {
  categoryForOperation,
  groupSchema,
} from "../../../../litellm/proxy/api_reference/route-categories.mjs";

test("should put inference and provider APIs in AI Routes", () => {
  for (const tag of [
    "chat/completions",
    "responses",
    "google genai endpoints",
    "WebSocket",
    "llm_passthrough",
    "OpenAI Pass-through",
    "anthropic_passthrough",
    "images",
    "audio",
    "files",
    "batch",
    "vector_stores",
    "a2a",
  ]) {
    assert.equal(
      categoryForOperation("/provider/action", { tags: [tag] }, "post"),
      "ai",
      tag,
    );
  }
});

test("should keep model discovery in AI and model configuration in Management", () => {
  const operation = { tags: ["model management"] };
  for (const path of [
    "/models",
    "/v1/models",
    "/models/{model_id}",
    "/v1/models/{model_id}",
  ]) {
    assert.equal(categoryForOperation(path, operation, "get"), "ai", path);
    assert.equal(
      categoryForOperation(path, operation, "post"),
      "management",
      path,
    );
  }
  assert.equal(
    categoryForOperation("/model/new", operation, "post"),
    "management",
  );
});

test("should include untagged MCP transports, MCP settings, and generic OAuth routes", () => {
  for (const path of [
    "/mcp",
    "/mcp/proxy",
    "/mcp-rest/tools/call",
    "/v1/mcp/server",
    "/toolset/{toolset_name}/mcp",
    "/{mcp_server_name}/mcp",
    "/get/mcp_semantic_filter_settings",
    "/update/mcp_tool_search_settings",
  ]) {
    assert.equal(categoryForOperation(path, {}, "post"), "mcp", path);
  }
  assert.equal(
    categoryForOperation("/token", { tags: ["mcp_discoverable"] }, "post"),
    "mcp",
  );
  assert.equal(
    categoryForOperation(
      "/public/mcp_hub",
      { tags: ["[beta] MCP", "public"] },
      "get",
    ),
    "mcp",
  );
});

test("should retain health, administration, and new unclassified operations in Management", () => {
  for (const [path, tags] of [
    ["/health", ["health"]],
    ["/key/generate", ["key management"]],
    ["/config/pass_through_endpoint", []],
    ["/brand-new-endpoint", ["unknown future tag"]],
  ])
    assert.equal(
      categoryForOperation(path, { tags }, "post"),
      "management",
      path,
    );
});

test("should partition methods without losing metadata, schemas, or authentication", () => {
  const schema = {
    openapi: "3.1.0",
    info: {
      title: "Gateway",
      version: "1",
      description: "Custom enterprise description",
    },
    servers: [{ url: "/deployment" }],
    security: [{ apiKey: [] }],
    components: {
      securitySchemes: {
        apiKey: { type: "apiKey", in: "header", name: "x-api-key" },
      },
    },
    tags: [
      { name: "model management", description: "Manage or discover models" },
      { name: "mcp_management", description: "MCP servers" },
      { name: "unused" },
    ],
    paths: {
      "/models": {
        parameters: [{ name: "tenant", in: "query" }],
        summary: "Available models",
        get: { tags: ["model management"], operationId: "listModels" },
        post: { tags: ["model management"], operationId: "createModel" },
      },
      "/chat/completions": {
        post: { tags: ["chat/completions"], operationId: "chat" },
      },
      "/token": { post: { tags: ["mcp_discoverable"], operationId: "oauth" } },
      "/mcp": {
        get: { operationId: "connect" },
        post: { operationId: "send" },
      },
      "/future": { patch: { operationId: "newFeature" } },
    },
  };
  const original = structuredClone(schema);
  const groups = groupSchema(schema);
  const byId = Object.fromEntries(groups.map((group) => [group.id, group]));
  assert.equal(byId.ai.schema.paths["/models"].get.operationId, "listModels");
  assert.equal(byId.ai.schema.paths["/models"].post, undefined);
  assert.equal(
    byId.management.schema.paths["/models"].post.operationId,
    "createModel",
  );
  assert.equal(byId.management.schema.paths["/models"].get, undefined);
  const operations = groups.flatMap((group) =>
    Object.values(group.schema.paths).flatMap((path) =>
      Object.values(path)
        .filter((value) => value.operationId)
        .map((value) => value.operationId),
    ),
  );
  assert.deepEqual(operations.sort(), [
    "chat",
    "connect",
    "createModel",
    "listModels",
    "newFeature",
    "oauth",
    "send",
  ]);
  assert.equal(
    groups.reduce((sum, group) => sum + group.count, 0),
    operations.length,
  );
  for (const group of groups) {
    for (const key of ["info", "servers", "security", "components"])
      assert.equal(group.schema[key], schema[key]);
    assert.ok(!group.schema.tags.some((tag) => tag.name === "unused"));
  }
  assert.equal(
    byId.ai.schema.paths["/models"].parameters,
    schema.paths["/models"].parameters,
  );
  assert.equal(
    byId.management.schema.paths["/models"].summary,
    "Available models",
  );
  assert.deepEqual(
    byId.ai.schema.tags.map((tag) => tag.name),
    ["chat/completions", "model management"],
  );
  assert.equal(byId.ai.schema.tags[1], schema.tags[0]);
  assert.deepEqual(schema, original);
});
