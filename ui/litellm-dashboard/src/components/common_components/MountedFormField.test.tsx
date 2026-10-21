import React from "react";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useForm } from "react-hook-form";
import type { UseFormGetValues } from "react-hook-form";

import {
  MountedFormField,
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFieldName,
  type MountedFormValues,
  type MountRegistry,
} from "./MountedFormField";

const registryOf = (names: readonly MountedFieldName[]): MountRegistry => ({
  register: () => () => undefined,
  mountedNames: () => names,
});

const getValuesOf = (store: Readonly<Record<string, unknown>>): UseFormGetValues<MountedFormValues> =>
  ((names: readonly string[]) => names.map((name) => store[name])) as unknown as UseFormGetValues<MountedFormValues>;

const project = (store: Readonly<Record<string, unknown>>) =>
  projectMountedValues(registryOf(Object.keys(store)), getValuesOf(store));

const projectPaths = (entries: readonly (readonly [MountedFieldName, unknown])[]) => {
  const store = Object.fromEntries(
    entries.map(([name, value]) => [Array.isArray(name) ? name.join(".") : (name as string), value]),
  );
  return projectMountedValues(registryOf(entries.map(([name]) => name)), getValuesOf(store));
};

describe("projectMountedValues", () => {
  it("keeps a flat name flat", () => {
    expect(project({ server_name: "s1", transport: "http" })).toStrictEqual({ server_name: "s1", transport: "http" });
  });

  it("nests an ARRAY name into a credentials object", () => {
    expect(
      projectPaths([
        [["credentials", "aws_region_name"], "us-east-1"],
        [["credentials", "aws_access_key_id"], "AKIA"],
      ]),
    ).toStrictEqual({ credentials: { aws_region_name: "us-east-1", aws_access_key_id: "AKIA" } });
  });

  it("keeps a literal dotted STRING name flat, matching antd getNamePath toArray", () => {
    expect(projectPaths([["a.b", 1]])).toStrictEqual({ "a.b": 1 });
    expect(projectPaths([["schema.property.with.dots", "v"]])).toStrictEqual({ "schema.property.with.dots": "v" });
  });

  it("rebuilds Form.List rows as an array, not an object keyed by digits", () => {
    const projected = projectPaths([
      [["env_vars", "0", "name"], "API_KEY"],
      [["env_vars", "0", "description"], "the key"],
      [["env_vars", "1", "name"], "REGION"],
    ]);
    expect(projected).toStrictEqual({
      env_vars: [{ name: "API_KEY", description: "the key" }, { name: "REGION" }],
    });
    expect(Array.isArray(projected.env_vars)).toBe(true);
  });

  it("rebuilds static_headers rows, the second Form.List site", () => {
    expect(
      projectPaths([
        [["static_headers", "0", "key"], "X-Tenant"],
        [["static_headers", "0", "value"], "acme"],
      ]),
    ).toStrictEqual({
      static_headers: [{ key: "X-Tenant", value: "acme" }],
    });
  });

  it("emits a mounted-but-unset field as a key holding undefined, matching antd onFinish", () => {
    const projected = project({ alias: undefined });
    expect(Object.keys(projected)).toStrictEqual(["alias"]);
    expect(projected.alias).toBeUndefined();
  });

  it("leaves a sparse row index as a hole rather than shifting later rows down", () => {
    const projected = projectPaths([[["env_vars", "2", "name"], "THIRD"]]) as { env_vars: readonly unknown[] };
    expect(projected.env_vars).toHaveLength(3);
    expect(projected.env_vars[2]).toStrictEqual({ name: "THIRD" });
  });

  it("mixes flat, nested and list names in one projection", () => {
    expect(
      projectPaths([
        ["transport", "http"],
        [["credentials", "client_id"], "cid"],
        [["env_vars", "0", "name"], "K"],
      ]),
    ).toStrictEqual({
      transport: "http",
      credentials: { client_id: "cid" },
      env_vars: [{ name: "K" }],
    });
  });
});

describe("useMountRegistry lifecycle", () => {
  const GatedForm: React.FC<{
    showOptional: boolean;
    showRequired: boolean;
    onFinish: (v: MountedFormValues) => void;
  }> = ({ showOptional, showRequired, onFinish }) => {
    const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: { server_name: "keep" } });
    const registry = useMountRegistry();
    return (
      <MountedFormProvider value={{ control: form.control, registry }}>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void form
              .trigger(registry.mountedNames().map((n) => (Array.isArray(n) ? n.join(".") : (n as string))))
              .then((valid) => {
                if (valid) onFinish(projectMountedValues(registry, form.getValues));
              });
          }}
        >
          <MountedFormField name="server_name">
            {(control) => (
              <input aria-label="server_name" value={String(control.value ?? "")} onChange={control.onChange} />
            )}
          </MountedFormField>
          {showOptional && (
            <MountedFormField name="alias">
              {(control) => (
                <input aria-label="alias" value={String(control.value ?? "")} onChange={control.onChange} />
              )}
            </MountedFormField>
          )}
          {showRequired && (
            <MountedFormField name="token_url" rules={{ required: "Token URL is required" }}>
              {(control) => (
                <input aria-label="token_url" value={String(control.value ?? "")} onChange={control.onChange} />
              )}
            </MountedFormField>
          )}
          <button type="submit">Submit</button>
        </form>
      </MountedFormProvider>
    );
  };

  it("drops a field's key from the submitted payload once its gate unmounts it", async () => {
    const onFinish = vi.fn();
    const { rerender } = render(<GatedForm showOptional showRequired={false} onFinish={onFinish} />);

    await userEvent.click(screen.getByRole("button", { name: "Submit" }));
    await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1));
    expect(Object.keys(onFinish.mock.calls[0][0] as object)).toContain("alias");

    rerender(<GatedForm showOptional={false} showRequired={false} onFinish={onFinish} />);
    await userEvent.click(screen.getByRole("button", { name: "Submit" }));
    await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(2));
    expect(Object.keys(onFinish.mock.calls[1][0] as object)).not.toContain("alias");
  });

  it("submits after a required field is unmounted, rather than validating a field the user can no longer see", async () => {
    const onFinish = vi.fn();
    const { rerender } = render(<GatedForm showOptional={false} showRequired onFinish={onFinish} />);

    await userEvent.click(screen.getByRole("button", { name: "Submit" }));
    expect(await screen.findByText("Token URL is required")).toBeInTheDocument();
    expect(onFinish).not.toHaveBeenCalled();

    rerender(<GatedForm showOptional={false} showRequired={false} onFinish={onFinish} />);
    await userEvent.click(screen.getByRole("button", { name: "Submit" }));
    await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1));
    expect(Object.keys(onFinish.mock.calls[0][0] as object)).not.toContain("token_url");
  });

  it("keeps a name mounted while a second field still holds a registration on it", () => {
    const registry = renderHook(() => useMountRegistry()).result.current;
    const releaseFirst = registry.register("credentials.scopes");
    registry.register("credentials.scopes");

    releaseFirst();

    expect(registry.mountedNames()).toStrictEqual(["credentials.scopes"]);
  });
});
