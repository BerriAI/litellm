import React from "react";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { useForm, type UseFormReturn } from "react-hook-form";

import {
  MountedFormField,
  MountedFormProvider,
  applyFieldValues,
  changedValuesFor,
  projectMountedValues,
  resetFieldsToDefaults,
  useMountRegistry,
  useMountedWatch,
  type MountRegistry,
  type MountedFormValues,
} from "./MountedFormField";

const harness: {
  form?: UseFormReturn<MountedFormValues>;
  registry?: MountRegistry;
} = {};

interface HarnessProps {
  readonly defaultValues: MountedFormValues;
  readonly showGated?: boolean;
  readonly showNested?: boolean;
  readonly showRows?: number;
  readonly duplicateGated?: boolean;
}

const Watcher: React.FC = () => {
  const gated = useMountedWatch("gated");
  const credentials = useMountedWatch("credentials");
  const rows = useMountedWatch("rows");
  return (
    <>
      <div data-testid="watch-gated">{JSON.stringify(gated) ?? "undefined"}</div>
      <div data-testid="watch-credentials">{JSON.stringify(credentials) ?? "undefined"}</div>
      <div data-testid="watch-rows">{JSON.stringify(rows) ?? "undefined"}</div>
    </>
  );
};

const Harness: React.FC<HarnessProps> = ({
  defaultValues,
  showGated = true,
  showNested = true,
  showRows = 0,
  duplicateGated = false,
}) => {
  const form = useForm<MountedFormValues>({ defaultValues });
  const registry = useMountRegistry();
  React.useEffect(() => {
    harness.form = form;
    harness.registry = registry;
  }, [form, registry]);
  return (
    <MountedFormProvider value={{ control: form.control, registry }}>
      <Watcher />
      <MountedFormField name="always" label="Always">
        {(field) => (
          <input aria-label="always" value={String(field.value ?? "")} onChange={field.onChange} id={field.id} />
        )}
      </MountedFormField>
      {showGated && (
        <MountedFormField name="gated" label="Gated">
          {(field) => (
            <input aria-label="gated" value={String(field.value ?? "")} onChange={field.onChange} id={field.id} />
          )}
        </MountedFormField>
      )}
      {duplicateGated && (
        <MountedFormField name="gated" label="Gated elsewhere">
          {(field) => (
            <input aria-label="gated-2" value={String(field.value ?? "")} onChange={field.onChange} id={field.id} />
          )}
        </MountedFormField>
      )}
      {showNested && (
        <MountedFormField name="credentials.client_id" label="Client id">
          {(field) => (
            <input aria-label="client_id" value={String(field.value ?? "")} onChange={field.onChange} id={field.id} />
          )}
        </MountedFormField>
      )}
      {Array.from({ length: showRows }, (_, index) => (
        <MountedFormField key={index} name={`rows.${index}.header`} label={`Header ${index}`}>
          {(field) => (
            <input
              aria-label={`header-${index}`}
              value={String(field.value ?? "")}
              onChange={field.onChange}
              id={field.id}
            />
          )}
        </MountedFormField>
      ))}
    </MountedFormProvider>
  );
};

const DEFAULTS: MountedFormValues = {
  always: "a",
  gated: "seeded",
  credentials: { client_id: "cid", access_token: "tok" },
  rows: [{ header: "h0" }, { header: "h1" }],
  never_bound: "leaked",
};

describe("projectMountedValues", () => {
  it("drops store keys that no field mounted, which is what keeps a spread payload out of the request", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    const projected = projectMountedValues(harness.registry!, harness.form!.getValues());
    expect(projected).not.toHaveProperty("never_bound");
    expect(harness.form!.getValues()).toHaveProperty("never_bound", "leaked");
  });

  it("rebuilds a container from only its mounted descendants", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    const projected = projectMountedValues(harness.registry!, harness.form!.getValues());
    expect(projected.credentials).toStrictEqual({ client_id: "cid" });
  });

  it("rebuilds an indexed path as an array, not an object", () => {
    render(<Harness defaultValues={DEFAULTS} showRows={2} />);
    const projected = projectMountedValues(harness.registry!, harness.form!.getValues());
    expect(projected.rows).toStrictEqual([{ header: "h0" }, { header: "h1" }]);
  });

  it("keeps a name mounted while a second field still binds it, so a shared name is not dropped early", () => {
    const { rerender } = render(<Harness defaultValues={DEFAULTS} duplicateGated />);
    rerender(<Harness defaultValues={DEFAULTS} duplicateGated showGated={false} />);
    expect(projectMountedValues(harness.registry!, harness.form!.getValues())).toHaveProperty("gated", "seeded");
  });

  it("accepts a getValues function as well as a plain store", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    expect(projectMountedValues(harness.registry!, harness.form!.getValues)).toStrictEqual(
      projectMountedValues(harness.registry!, harness.form!.getValues()),
    );
  });

  it("omits a gated field once it unmounts", () => {
    const { rerender } = render(<Harness defaultValues={DEFAULTS} />);
    expect(projectMountedValues(harness.registry!, harness.form!.getValues())).toHaveProperty("gated");
    rerender(<Harness defaultValues={DEFAULTS} showGated={false} />);
    expect(projectMountedValues(harness.registry!, harness.form!.getValues())).not.toHaveProperty("gated");
  });
});

describe("useMountedWatch", () => {
  it("is undefined for a seeded field that never mounted, so a `watched ?? saved` fallback still reads the saved value", () => {
    render(<Harness defaultValues={DEFAULTS} showGated={false} />);
    expect(screen.getByTestId("watch-gated")).toHaveTextContent("undefined");
  });

  it("reports the live value while the field is mounted", async () => {
    render(<Harness defaultValues={DEFAULTS} />);
    await userEvent.clear(screen.getByLabelText("gated"));
    await userEvent.type(screen.getByLabelText("gated"), "typed");
    expect(screen.getByTestId("watch-gated")).toHaveTextContent('"typed"');
  });

  it("goes back to undefined after the field unmounts even though the store keeps the value", async () => {
    const { rerender } = render(<Harness defaultValues={DEFAULTS} />);
    await userEvent.clear(screen.getByLabelText("gated"));
    await userEvent.type(screen.getByLabelText("gated"), "typed");
    rerender(<Harness defaultValues={DEFAULTS} showGated={false} />);
    expect(screen.getByTestId("watch-gated")).toHaveTextContent("undefined");
    expect(harness.form!.getValues("gated")).toBe("typed");
  });

  it("narrows a container to its mounted descendants", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    expect(screen.getByTestId("watch-credentials")).toHaveTextContent('{"client_id":"cid"}');
  });

  it("is undefined for a container whose descendants all unmounted", () => {
    render(<Harness defaultValues={DEFAULTS} showNested={false} />);
    expect(screen.getByTestId("watch-credentials")).toHaveTextContent("undefined");
  });
});

describe("applyFieldValues", () => {
  it("deep-merges a partial object instead of replacing it", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    act(() => applyFieldValues(harness.form!, { credentials: { client_id: "next" } }));
    expect(harness.form!.getValues("credentials")).toStrictEqual({ client_id: "next", access_token: "tok" });
  });

  it("replaces arrays rather than merging them index by index", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    act(() => applyFieldValues(harness.form!, { rows: [{ header: "only" }] }));
    expect(harness.form!.getValues("rows")).toStrictEqual([{ header: "only" }]);
  });

  it("clears a key when the patch carries undefined", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    act(() => applyFieldValues(harness.form!, { credentials: undefined }));
    expect(harness.form!.getValues("credentials")).toBeUndefined();
  });
});

describe("resetFieldsToDefaults", () => {
  it("restores a container path that has no field registered under that exact name", () => {
    render(<Harness defaultValues={DEFAULTS} />);
    act(() => harness.form!.setValue("credentials", { client_id: "dirty", minted: "token" }));
    act(() => resetFieldsToDefaults(harness.form!, DEFAULTS, ["credentials"]));
    expect(harness.form!.getValues("credentials")).toStrictEqual({ client_id: "cid", access_token: "tok" });
  });

  it("restores a path whose field is not mounted at all", () => {
    render(<Harness defaultValues={DEFAULTS} showGated={false} />);
    act(() => harness.form!.setValue("gated", "dirty"));
    act(() => resetFieldsToDefaults(harness.form!, DEFAULTS, ["gated"]));
    expect(harness.form!.getValues("gated")).toBe("seeded");
  });
});

describe("changedValuesFor", () => {
  it("nests a dotted path the way an antd onValuesChange payload is shaped", () => {
    expect(changedValuesFor("credentials.client_id", { credentials: { client_id: "x", other: "y" } })).toStrictEqual({
      credentials: { client_id: "x" },
    });
  });

  it("keeps a top-level key flat so `key in changedValues` still answers", () => {
    expect("url" in changedValuesFor("url", { url: "https://example.com" })).toBe(true);
  });
});
