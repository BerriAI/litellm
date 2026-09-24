import { describe, it, expect } from "vitest";
import {
  buildCoordinationPayload,
  buildInitialValues,
  configuredSecretFields,
  fieldsForSection,
  inferRedisType,
  sourceBadge,
} from "./coordinationRedisUtils";
import { REDACTED_VALUE } from "./types";

describe("fieldsForSection", () => {
  it("should only include a redis-type-specific field when that type is selected", () => {
    expect(fieldsForSection("cluster", "cluster").map((f) => f.name)).toEqual(["startup_nodes"]);
    expect(fieldsForSection("cluster", "node")).toEqual([]);
    expect(fieldsForSection("sentinel", "sentinel").map((f) => f.name)).toEqual([
      "sentinel_nodes",
      "service_name",
      "sentinel_password",
    ]);
  });

  it("should include connection fields for every redis type in schema order", () => {
    expect(fieldsForSection("connection", "sentinel").map((f) => f.name)).toEqual([
      "url",
      "host",
      "port",
      "username",
      "password",
    ]);
  });
});

describe("inferRedisType", () => {
  it("should infer sentinel when sentinel nodes are configured", () => {
    expect(inferRedisType({ sentinel_nodes: [["localhost", 26379]] })).toBe("sentinel");
  });

  it("should infer cluster when startup nodes are configured", () => {
    expect(inferRedisType({ startup_nodes: [{ host: "127.0.0.1", port: 7001 }] })).toBe("cluster");
  });

  it("should infer node when neither cluster nor sentinel nodes are configured", () => {
    expect(inferRedisType({ host: "localhost" })).toBe("node");
    expect(inferRedisType({ startup_nodes: [], sentinel_nodes: [] })).toBe("node");
  });
});

describe("buildInitialValues", () => {
  it("should apply defaults as strings for text inputs and coerce booleans", () => {
    const values = buildInitialValues({});
    expect(values.port).toBe("6379");
    expect(values.ssl).toBe(false);
    expect(values.host).toBe("");
  });

  it("should never load a redacted secret into the form, so typing cannot append to the marker", () => {
    const values = buildInitialValues({ password: REDACTED_VALUE, url: REDACTED_VALUE });
    expect(values.password).toBe("");
    expect(values.url).toBe("");
  });

  it("should stringify list values so they render in a textarea", () => {
    const nodes = [{ host: "127.0.0.1", port: 7001 }];
    const values = buildInitialValues({ startup_nodes: nodes });
    expect(values.startup_nodes).toBe(JSON.stringify(nodes, null, 2));
  });
});

describe("configuredSecretFields", () => {
  it("should report which secrets the backend already holds so the form can say so", () => {
    expect(configuredSecretFields({ password: REDACTED_VALUE, host: "localhost" })).toEqual(new Set(["password"]));
  });

  it("should not report an unset secret", () => {
    expect(configuredSecretFields({ password: "", sentinel_password: null })).toEqual(new Set());
  });
});

describe("buildCoordinationPayload", () => {
  it("should drop empty fields and send the port as a number", () => {
    const payload = buildCoordinationPayload("node", { host: "localhost", port: "6379", username: "" });
    expect(payload).toEqual({ host: "localhost", port: 6379, ssl: false });
    expect(payload).not.toHaveProperty("username");
  });

  it("should not resubmit a secret that is still the redacted marker", () => {
    const untouchedSecrets = {
      host: "localhost",
      password: REDACTED_VALUE,
      url: REDACTED_VALUE,
      sentinel_password: REDACTED_VALUE,
    };
    const payload = buildCoordinationPayload("sentinel", untouchedSecrets);
    expect(payload).not.toHaveProperty("password");
    expect(payload).not.toHaveProperty("url");
    expect(payload).not.toHaveProperty("sentinel_password");
  });

  it("should submit a secret once the admin replaces the redacted marker", () => {
    const payload = buildCoordinationPayload("node", { password: "hunter2" });
    expect(payload.password).toBe("hunter2");
  });

  it("should parse cluster startup nodes from their textarea string into an array", () => {
    const payload = buildCoordinationPayload("cluster", {
      startup_nodes: '[{"host":"127.0.0.1","port":7001}]',
    });
    expect(payload.startup_nodes).toEqual([{ host: "127.0.0.1", port: 7001 }]);
  });

  it("should parse sentinel nodes from their textarea string into an array of pairs", () => {
    const payload = buildCoordinationPayload("sentinel", {
      sentinel_nodes: '[["localhost", 26379]]',
      service_name: "mymaster",
    });
    expect(payload.sentinel_nodes).toEqual([["localhost", 26379]]);
    expect(payload.service_name).toBe("mymaster");
  });

  it("should omit a list field whose textarea holds invalid JSON", () => {
    const payload = buildCoordinationPayload("cluster", { startup_nodes: "not json" });
    expect(payload).not.toHaveProperty("startup_nodes");
  });

  it("should exclude fields that do not belong to the selected redis type", () => {
    const payload = buildCoordinationPayload("node", {
      sentinel_nodes: '[["localhost",26379]]',
      startup_nodes: '[{"host":"127.0.0.1","port":7001}]',
    });
    expect(payload).not.toHaveProperty("sentinel_nodes");
    expect(payload).not.toHaveProperty("startup_nodes");
  });
});

describe("sourceBadge", () => {
  it("should label each backend source value", () => {
    expect(sourceBadge("coordination_redis").label).toBe("Configured here");
    expect(sourceBadge("cache_backend").label).toBe("Borrowed from response cache");
    expect(sourceBadge("environment").label).toBe("From REDIS_* environment");
    expect(sourceBadge(null).label).toBe("Not configured");
  });

  it("should tone only a dedicated coordination Redis as success", () => {
    expect(sourceBadge("coordination_redis").tone).toBe("success");
    expect(sourceBadge("cache_backend").tone).toBe("info");
    expect(sourceBadge("environment").tone).toBe("info");
    expect(sourceBadge(null).tone).toBe("neutral");
  });

  it("should fall back to not configured for an unrecognized source", () => {
    expect(sourceBadge("something_new").label).toBe("Not configured");
  });
});
