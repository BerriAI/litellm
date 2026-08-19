import { describe, expect, it } from "vitest";
import type { UseFormGetValues } from "react-hook-form";

import { projectMountedValues, type MountedFormValues, type MountRegistry } from "./MountedFormField";

const registryOf = (names: readonly string[]): MountRegistry => ({
  register: () => () => undefined,
  mountedNames: () => names,
});

const getValuesOf = (store: Readonly<Record<string, unknown>>): UseFormGetValues<MountedFormValues> =>
  ((names: readonly string[]) => names.map((name) => store[name])) as unknown as UseFormGetValues<MountedFormValues>;

const project = (store: Readonly<Record<string, unknown>>) =>
  projectMountedValues(registryOf(Object.keys(store)), getValuesOf(store));

describe("projectMountedValues", () => {
  it("keeps a flat name flat", () => {
    expect(project({ server_name: "s1", transport: "http" })).toStrictEqual({ server_name: "s1", transport: "http" });
  });

  it("rebuilds a nested credentials object rather than emitting a dotted key", () => {
    expect(
      project({ "credentials.aws_region_name": "us-east-1", "credentials.aws_access_key_id": "AKIA" }),
    ).toStrictEqual({ credentials: { aws_region_name: "us-east-1", aws_access_key_id: "AKIA" } });
  });

  it("rebuilds Form.List rows as an array, not an object keyed by digits", () => {
    const projected = project({
      "env_vars.0.name": "API_KEY",
      "env_vars.0.description": "the key",
      "env_vars.1.name": "REGION",
    });
    expect(projected).toStrictEqual({
      env_vars: [{ name: "API_KEY", description: "the key" }, { name: "REGION" }],
    });
    expect(Array.isArray(projected.env_vars)).toBe(true);
  });

  it("rebuilds static_headers rows, the second Form.List site", () => {
    expect(project({ "static_headers.0.key": "X-Tenant", "static_headers.0.value": "acme" })).toStrictEqual({
      static_headers: [{ key: "X-Tenant", value: "acme" }],
    });
  });

  it("emits a mounted-but-unset field as a key holding undefined, matching antd onFinish", () => {
    const projected = project({ alias: undefined });
    expect(Object.keys(projected)).toStrictEqual(["alias"]);
    expect(projected.alias).toBeUndefined();
  });

  it("leaves a sparse row index as a hole rather than shifting later rows down", () => {
    const projected = project({ "env_vars.2.name": "THIRD" }) as { env_vars: readonly unknown[] };
    expect(projected.env_vars).toHaveLength(3);
    expect(projected.env_vars[2]).toStrictEqual({ name: "THIRD" });
  });

  it("mixes flat, nested and list names in one projection", () => {
    expect(project({ transport: "http", "credentials.client_id": "cid", "env_vars.0.name": "K" })).toStrictEqual({
      transport: "http",
      credentials: { client_id: "cid" },
      env_vars: [{ name: "K" }],
    });
  });
});
