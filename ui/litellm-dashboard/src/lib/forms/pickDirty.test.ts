import { act, renderHook } from "@testing-library/react";
import type { FieldValues, FormState } from "react-hook-form";
import { useFieldArray, useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";

import { pickDirty } from "./pickDirty";

const dirty = <T extends FieldValues>(map: Record<string, unknown>) => map as FormState<T>["dirtyFields"];

describe("pickDirty", () => {
  it("omits untouched keys entirely rather than sending them as undefined", () => {
    const result = pickDirty({ team_alias: "a", tpm_limit: 5 }, dirty({ team_alias: true }));

    expect(result).toEqual({ team_alias: "a" });
    expect("tpm_limit" in result).toBe(false);
  });

  it("returns an empty patch when nothing is dirty", () => {
    expect(pickDirty({ team_alias: "a", models: ["gpt-4"] }, dirty({}))).toEqual({});
  });

  describe("clear tokens survive", () => {
    it.each([
      ["null scalar", null],
      ["empty string", ""],
      ["zero", 0],
      ["false", false],
    ])("keeps a dirty key whose value is %s", (_label, value) => {
      const result = pickDirty({ max_budget: value }, dirty({ max_budget: true }));

      expect("max_budget" in result).toBe(true);
      expect(result.max_budget).toBe(value);
    });

    it("keeps a dirty empty array, which is how lists are cleared", () => {
      expect(pickDirty({ models: [] }, dirty({ models: true }))).toEqual({ models: [] });
    });

    it("keeps a dirty empty object, which is how model_aliases is cleared", () => {
      expect(pickDirty({ model_aliases: {} }, dirty({ model_aliases: true }))).toEqual({ model_aliases: {} });
    });
  });

  describe("dirtiness is read at the top level", () => {
    it("sends the whole array when any element is dirty", () => {
      const values = { models: ["gpt-4", "gpt-5", "opus"] };

      expect(pickDirty(values, dirty({ models: [false, true, false] }))).toEqual(values);
    });

    it("omits the array when no element is dirty", () => {
      const result = pickDirty({ models: ["gpt-4"] }, dirty({ models: [false, false] }));

      expect("models" in result).toBe(false);
    });

    it("sends the whole object when one nested leaf is dirty", () => {
      const values = { object_permission: { vector_stores: ["vs-1"], agents: ["a-1"] } };

      expect(pickDirty(values, dirty({ object_permission: { vector_stores: true, agents: false } }))).toEqual(values);
    });

    it("tolerates the null holes RHF leaves in sparse per-leaf dirty arrays", () => {
      const values = {
        modelLimits: [
          { model: "gpt-4", tpm: 1 },
          { model: "opus", tpm: 2 },
        ],
      };

      expect(pickDirty(values, dirty({ modelLimits: [null, { tpm: true }] }))).toEqual(values);
    });

    it("omits an object whose every nested leaf is clean", () => {
      const result = pickDirty(
        { object_permission: { vector_stores: ["vs-1"] } },
        dirty({ object_permission: { vector_stores: false } }),
      );

      expect("object_permission" in result).toBe(false);
    });
  });

  it("ignores dirty keys that are absent from the submitted values", () => {
    const result = pickDirty({ team_alias: "a" }, dirty({ team_alias: true, ghost_field: true }));

    expect(result).toEqual({ team_alias: "a" });
    expect("ghost_field" in result).toBe(false);
  });

  it("does not mutate its inputs", () => {
    const values = { models: ["gpt-4"], tpm_limit: 5 };
    const dirtyFields = dirty<typeof values>({ models: [true] });

    pickDirty(values, dirtyFields);

    expect(values).toEqual({ models: ["gpt-4"], tpm_limit: 5 });
    expect(dirtyFields).toEqual({ models: [true] });
  });

  it("keeps value identity so nested references are not cloned", () => {
    const models = ["gpt-4"];

    expect(pickDirty({ models }, dirty({ models: true })).models).toBe(models);
  });
});

describe("pickDirty against a real react-hook-form instance", () => {
  const defaultValues = {
    team_alias: "team-a",
    max_budget: 10 as number | null,
    models: ["gpt-4", "opus"] as string[],
    object_permission: { vector_stores: ["vs-1"] as string[] },
    modelLimits: [{ model: "gpt-4", tpm: 1 }],
  };

  const renderForm = () =>
    renderHook(() => {
      const form = useForm({ defaultValues });
      const fieldArray = useFieldArray({ control: form.control, name: "modelLimits" });
      void form.formState.dirtyFields;
      return { form, fieldArray };
    });

  const patchOf = (result: { current: { form: ReturnType<typeof useForm<typeof defaultValues>> } }) =>
    pickDirty(result.current.form.getValues(), result.current.form.formState.dirtyFields);

  it("sends nothing when the user opens the form and saves without editing", () => {
    const { result } = renderForm();

    expect(patchOf(result)).toEqual({});
  });

  it("sends only the edited scalar", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({ team_alias: "team-b" });
  });

  it("sends null to clear a scalar, and nothing else", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("max_budget", null, { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({ max_budget: null });
  });

  it("sends an empty array to clear a list emptied through useFieldArray", () => {
    const { result } = renderForm();

    act(() => {
      result.current.fieldArray.remove(0);
    });

    expect(patchOf(result)).toEqual({ modelLimits: [] });
  });

  it("sends the whole nested object when one leaf under it changes", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("object_permission.vector_stores", [], { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({ object_permission: { vector_stores: [] } });
  });

  it("drops a field the user edited and then reverted to its original value", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });
    act(() => {
      result.current.form.setValue("team_alias", "team-a", { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({});
  });

  it("resets to a clean baseline after a successful save", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });
    act(() => {
      result.current.form.reset({ ...defaultValues, team_alias: "team-b" });
    });

    expect(patchOf(result)).toEqual({});
  });
});
