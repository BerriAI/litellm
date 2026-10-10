import React from "react";
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { useForm } from "react-hook-form";

import type { MountedFormValues } from "@/components/common_components/MountedFormField";
import { allFieldsValue, deepMergedFieldsValue, resetFields, setFieldsValue, singleBranchChange } from "./mcpFormStore";

const withForm = (
  defaultValues: MountedFormValues,
  act: (form: ReturnType<typeof useForm<MountedFormValues>>) => void,
) => {
  let store: MountedFormValues = {};
  const Probe: React.FC = () => {
    const form = useForm<MountedFormValues>({ defaultValues });
    React.useEffect(() => {
      act(form);
      store = allFieldsValue(form);
    }, [form]);
    return null;
  };
  render(<Probe />);
  return store;
};

describe("deepMergedFieldsValue", () => {
  it("keeps a sibling key when a nested object is written, which is what preserves a declared app", () => {
    expect(
      deepMergedFieldsValue(
        { credentials: { client_id: "kept", access_token: "tok" } },
        { credentials: { client_id: "typed" } },
      ),
    ).toStrictEqual({ credentials: { client_id: "typed", access_token: "tok" } });
  });

  it("replaces an array rather than merging it index by index", () => {
    expect(deepMergedFieldsValue({ extra_headers: ["a", "b", "c"] }, { extra_headers: ["z"] })).toStrictEqual({
      extra_headers: ["z"],
    });
  });

  it("writes an explicit undefined instead of skipping the key, which is how a transport switch clears a field", () => {
    const merged = deepMergedFieldsValue({ url: "https://old", auth_type: "api_key" }, { url: undefined });
    expect(merged).toStrictEqual({ url: undefined, auth_type: "api_key" });
    expect("url" in merged).toBe(true);
  });

  it("writes an explicit null rather than treating it as a merge target", () => {
    expect(deepMergedFieldsValue({ credentials: { client_id: "x" } }, { credentials: null })).toStrictEqual({
      credentials: null,
    });
  });

  it("replaces a primitive with an object when the incoming value is an object", () => {
    expect(deepMergedFieldsValue({ credentials: "not-an-object" }, { credentials: { client_id: "x" } })).toStrictEqual({
      credentials: { client_id: "x" },
    });
  });

  it("does not mutate the store it was handed", () => {
    const store = { credentials: { client_id: "kept" } };
    deepMergedFieldsValue(store, { credentials: { client_secret: "added" } });
    expect(store).toStrictEqual({ credentials: { client_id: "kept" } });
  });

  it("treats a missing store as empty rather than throwing", () => {
    expect(deepMergedFieldsValue(undefined, { alias: "a" })).toStrictEqual({ alias: "a" });
  });
});

describe("singleBranchChange", () => {
  it("carries only the changed leaf, so re-applying it cannot resurrect a sibling token key", () => {
    expect(
      singleBranchChange("credentials.client_id", { credentials: { client_id: "typed", access_token: "stale" } }),
    ).toStrictEqual({ credentials: { client_id: "typed" } });
  });

  it("exposes the changed top-level key so an upstream-field check can test membership", () => {
    const changed = singleBranchChange("url", { url: "https://new", alias: "a" });
    expect("url" in changed).toBe(true);
    expect("alias" in changed).toBe(false);
  });

  it("builds an array for a numeric segment so a list row does not become an object keyed by index", () => {
    expect(
      singleBranchChange("static_headers.1.value", { static_headers: [{ value: "a" }, { value: "b" }] }),
    ).toStrictEqual({ static_headers: [undefined, { value: "b" }] });
  });

  it("yields an undefined leaf rather than throwing when the path is not in the store", () => {
    expect(singleBranchChange("credentials.client_secret", {})).toStrictEqual({
      credentials: { client_secret: undefined },
    });
  });
});

describe("resetFields", () => {
  it("restores the seeded value rather than clearing the key, so an edit reset keeps the saved server's credentials", () => {
    const store = withForm({ credentials: { client_id: "saved", access_token: "tok" } }, (form) => {
      form.setValue("credentials", { client_id: "typed" });
      resetFields(form, ["credentials"], { credentials: { client_id: "saved", access_token: "tok" } });
    });

    expect(store.credentials).toStrictEqual({ client_id: "saved", access_token: "tok" });
  });

  it("clears the key when no seed is supplied, which is what the create form's blank store means", () => {
    const store = withForm({ credentials: { client_id: "typed" } }, (form) => {
      resetFields(form, ["credentials"]);
    });

    expect(store).toHaveProperty("credentials", undefined);
  });
});

describe("setFieldsValue", () => {
  it("writes an undefined leaf into the live store, so a transport switch really clears the field", () => {
    const store = withForm({ url: "https://example.com", command: "npx" }, (form) => {
      setFieldsValue(form, { url: undefined });
    });

    expect(store).toStrictEqual({ url: undefined, command: "npx" });
  });

  it("merges a nested write into the live store instead of replacing the whole object", () => {
    const store = withForm({ credentials: { client_id: "kept", scopes: ["a"] } }, (form) => {
      setFieldsValue(form, { credentials: { client_secret: "new" } });
    });

    expect(store.credentials).toStrictEqual({ client_id: "kept", scopes: ["a"], client_secret: "new" });
  });
});
