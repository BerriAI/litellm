import { describe, it, expect } from "vitest";
import { buildCachePayload, buildInitialValues, fieldsForSection } from "./cacheSettingsUtils";

describe("fieldsForSection", () => {
  it("should only include a redis-type-specific field when that type is selected", () => {
    expect(fieldsForSection("cluster", "cluster", false).map((f) => f.name)).toEqual(["redis_startup_nodes"]);
    expect(fieldsForSection("cluster", "node", false)).toEqual([]);
  });

  it("should include semantic fields only when semanticEnabled is true", () => {
    expect(fieldsForSection("semantic", "node", false)).toEqual([]);
    expect(fieldsForSection("semantic", "node", true).map((f) => f.name)).toEqual([
      "similarity_threshold",
      "redis_semantic_cache_embedding_model",
    ]);
  });

  it("should include connection fields for every redis type in schema order", () => {
    expect(fieldsForSection("connection", "node", false).map((f) => f.name)).toEqual([
      "url",
      "host",
      "port",
      "db",
      "password",
      "username",
    ]);
  });
});

describe("buildInitialValues", () => {
  it("should apply defaults as strings for text inputs and coerce booleans", () => {
    const values = buildInitialValues({});
    expect(values.port).toBe("6379");
    expect(values.similarity_threshold).toBe("0.8");
    expect(values.ssl).toBe(false);
    expect(values.db).toBe("");
  });

  it("should stringify list values so they render in a textarea", () => {
    const nodes = [{ host: "127.0.0.1", port: "7001" }];
    const values = buildInitialValues({ redis_startup_nodes: nodes });
    expect(values.redis_startup_nodes).toBe(JSON.stringify(nodes, null, 2));
  });

  it("should render numeric current values as strings for their text inputs", () => {
    const values = buildInitialValues({ max_connections: 10 });
    expect(values.max_connections).toBe("10");
  });
});

describe("buildCachePayload", () => {
  it("should tag the payload as redis and drop empty fields and the UI-only redis_type", () => {
    const payload = buildCachePayload("node", { host: "localhost", port: "6379", username: "" }, false, {
      forTesting: false,
    });
    expect(payload).toEqual({
      type: "redis",
      host: "localhost",
      port: "6379",
      ssl: false,
      ssl_check_hostname: false,
    });
    expect(payload).not.toHaveProperty("redis_type");
    expect(payload).not.toHaveProperty("username");
  });

  it("should parse list fields from their textarea string into arrays", () => {
    const payload = buildCachePayload(
      "cluster",
      { redis_startup_nodes: '[{"host":"127.0.0.1","port":"7001"}]' },
      false,
      { forTesting: false },
    );
    expect(payload.redis_startup_nodes).toEqual([{ host: "127.0.0.1", port: "7001" }]);
  });

  it("should omit a list field whose textarea holds invalid JSON", () => {
    const payload = buildCachePayload("cluster", { redis_startup_nodes: "not json" }, false, { forTesting: false });
    expect(payload).not.toHaveProperty("redis_startup_nodes");
  });

  it("should send type redis-semantic when saving with semantic caching enabled", () => {
    const payload = buildCachePayload("node", { similarity_threshold: 0.9 }, true, { forTesting: false });
    expect(payload.type).toBe("redis-semantic");
    expect(payload.similarity_threshold).toBe(0.9);
  });

  it("should keep type redis when testing with semantic caching enabled so the test endpoint accepts it", () => {
    const payload = buildCachePayload("node", { similarity_threshold: 0.9 }, true, { forTesting: true });
    expect(payload.type).toBe("redis");
  });

  it("should exclude fields that do not belong to the selected redis type", () => {
    const payload = buildCachePayload("node", { sentinel_nodes: '[["localhost",26379]]' }, false, {
      forTesting: false,
    });
    expect(payload).not.toHaveProperty("sentinel_nodes");
  });
});
