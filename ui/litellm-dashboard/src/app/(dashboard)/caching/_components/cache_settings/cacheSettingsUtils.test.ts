import { describe, it, expect } from "vitest";
import { buildCachePayload, buildInitialValues, configuredSecretFields, fieldsForSection } from "./cacheSettingsUtils";
import { REDACTED_VALUE } from "./cacheSettingsFields";

describe("fieldsForSection", () => {
  it("should only include a redis-type-specific field when that type is selected", () => {
    expect(fieldsForSection("cluster", "cluster").map((f) => f.name)).toEqual(["redis_startup_nodes"]);
    expect(fieldsForSection("cluster", "node")).toEqual([]);
  });

  it("should include connection fields for every redis type in schema order", () => {
    expect(fieldsForSection("connection", "node").map((f) => f.name)).toEqual([
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
    const payload = buildCachePayload("node", { host: "localhost", port: "6379", username: "" }, { forTesting: false });
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
      { forTesting: false },
    );
    expect(payload.redis_startup_nodes).toEqual([{ host: "127.0.0.1", port: "7001" }]);
  });

  it("should omit a list field whose textarea holds invalid JSON", () => {
    const payload = buildCachePayload("cluster", { redis_startup_nodes: "not json" }, { forTesting: false });
    expect(payload).not.toHaveProperty("redis_startup_nodes");
  });

  it("should send type redis-semantic when saving a semantic cache", () => {
    const payload = buildCachePayload("semantic", { similarity_threshold: 0.9 }, { forTesting: false });
    expect(payload.type).toBe("redis-semantic");
    expect(payload.similarity_threshold).toBe(0.9);
  });

  it("should keep type redis when testing a semantic cache so the test endpoint accepts it", () => {
    const payload = buildCachePayload("semantic", { similarity_threshold: 0.9 }, { forTesting: true });
    expect(payload.type).toBe("redis");
  });

  it("should exclude fields that do not belong to the selected redis type", () => {
    const payload = buildCachePayload("node", { sentinel_nodes: '[["localhost",26379]]' }, { forTesting: false });
    expect(payload).not.toHaveProperty("sentinel_nodes");
  });

  it("should drop a secret whose value is the redacted marker so it is never persisted", () => {
    const payload = buildCachePayload(
      "node",
      { host: "localhost", password: REDACTED_VALUE, url: REDACTED_VALUE },
      { forTesting: false },
    );
    expect(payload).not.toHaveProperty("password");
    expect(payload).not.toHaveProperty("url");
    expect(payload.host).toBe("localhost");
  });

  it("should send a real new secret value the admin typed", () => {
    const payload = buildCachePayload("node", { password: "brandnewpw" }, { forTesting: false });
    expect(payload.password).toBe("brandnewpw");
  });
});

describe("secret handling", () => {
  it("buildInitialValues never prefills a credential, even when the server reports it configured", () => {
    const serverValues = {
      host: "localhost",
      password: REDACTED_VALUE,
      url: REDACTED_VALUE,
      sentinel_password: REDACTED_VALUE,
    };
    const values = buildInitialValues(serverValues);
    expect(values.password).toBe("");
    expect(values.url).toBe("");
    expect(values.sentinel_password).toBe("");
    // non-secret fields are still prefilled
    expect(values.host).toBe("localhost");
  });

  it("configuredSecretFields reports which credentials the server marked as set", () => {
    const configured = configuredSecretFields({
      password: REDACTED_VALUE,
      url: "",
      host: "localhost",
    });
    expect(configured.has("password")).toBe(true);
    expect(configured.has("url")).toBe(false);
    // a non-secret field is never reported as a configured secret
    expect(configured.has("host")).toBe(false);
  });
});
