import { describe, expect, it } from "vitest";

import {
  readWorkloadClass,
  stripWorkloadClass,
  withWorkloadClass,
  withWorkloadClassJson,
  workloadClassMetadata,
} from "./workloadClass";

describe("readWorkloadClass", () => {
  it("reads the named class from metadata.priority", () => {
    expect(readWorkloadClass({ priority: "batch", region: "us" })).toBe("batch");
  });

  it("falls back to the default pool when priority is missing, empty, or not a string", () => {
    expect(readWorkloadClass(undefined)).toBe("default");
    expect(readWorkloadClass(null)).toBe("default");
    expect(readWorkloadClass({})).toBe("default");
    expect(readWorkloadClass({ priority: "" })).toBe("default");
    expect(readWorkloadClass({ priority: 3 })).toBe("default");
  });
});

describe("workloadClassMetadata", () => {
  it("writes a named class under metadata.priority", () => {
    expect(workloadClassMetadata("production")).toEqual({ priority: "production" });
  });

  it("writes nothing for the default pool or an unset selection", () => {
    expect(workloadClassMetadata("default")).toEqual({});
    expect(workloadClassMetadata(undefined)).toEqual({});
    expect(workloadClassMetadata("")).toEqual({});
  });
});

describe("stripWorkloadClass", () => {
  it("drops only the priority key", () => {
    expect(stripWorkloadClass({ priority: "batch", team: "x", logging: [] })).toEqual({ team: "x", logging: [] });
  });

  it("passes through non-object metadata untouched", () => {
    expect(stripWorkloadClass(null)).toBeNull();
    expect(stripWorkloadClass(["a"])).toEqual(["a"]);
  });
});

describe("withWorkloadClass", () => {
  it("leaves metadata untouched when no selection was made, so a hand-typed priority survives", () => {
    expect(withWorkloadClass({ priority: "typed", region: "us" }, undefined)).toEqual({
      priority: "typed",
      region: "us",
    });
  });

  it("replaces priority for a named selection and removes it for the default pool", () => {
    expect(withWorkloadClass({ priority: "old", region: "us" }, "batch")).toEqual({ region: "us", priority: "batch" });
    expect(withWorkloadClass({ priority: "old", region: "us" }, "default")).toEqual({ region: "us" });
  });

  it("does not spread arrays or primitives into objects", () => {
    expect(withWorkloadClass(["a"], "batch")).toEqual(["a"]);
    expect(withWorkloadClass(5, "batch")).toBe(5);
  });
});

describe("withWorkloadClassJson", () => {
  it("replaces an old priority while keeping unrelated keys", () => {
    const out = withWorkloadClassJson('{"priority": "batch", "region": "us"}', "production");
    expect(JSON.parse(out ?? "")).toEqual({ region: "us", priority: "production" });
  });

  it("removes priority when the default pool is selected", () => {
    const out = withWorkloadClassJson('{"priority": "batch", "region": "us"}', "default");
    expect(JSON.parse(out ?? "")).toEqual({ region: "us" });
  });

  it("treats blank or missing metadata as an empty object", () => {
    expect(JSON.parse(withWorkloadClassJson(undefined, "batch") ?? "")).toEqual({ priority: "batch" });
    expect(withWorkloadClassJson("   ", "default")).toBe("   ");
  });

  it("returns the operator's text verbatim when the class did not change", () => {
    expect(withWorkloadClassJson('{ "region": "us" }', "default")).toBe('{ "region": "us" }');
    expect(withWorkloadClassJson('{"priority": "batch"}', "batch")).toBe('{"priority": "batch"}');
  });

  it("leaves invalid or non-object JSON alone so the form validator reports it", () => {
    expect(withWorkloadClassJson("{not json", "batch")).toBe("{not json");
    expect(withWorkloadClassJson("[1,2]", "batch")).toBe("[1,2]");
  });
});
