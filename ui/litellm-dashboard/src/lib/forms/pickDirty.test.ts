import { act, renderHook } from "@testing-library/react";
import { useFieldArray, useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";

import { usePickDirty } from "./pickDirty";

const defaultValues = {
  team_alias: "team-a",
  max_budget: 10 as number | null,
  models: ["gpt-4", "opus"] as string[],
  model_aliases: { fast: "gpt-4" } as Record<string, string>,
  object_permission: { vector_stores: ["vs-1"] as string[] },
  modelLimits: [{ model: "gpt-4", tpm: 1 }],
};

// Nothing here reads formState during render on purpose: the hook must own the dirty-state subscription itself
const renderForm = () =>
  renderHook(() => {
    const form = useForm({ defaultValues });
    const fieldArray = useFieldArray({ control: form.control, name: "modelLimits" });
    const pickDirty = usePickDirty(form.control);
    return { form, fieldArray, pickDirty };
  });

const patchOf = (result: ReturnType<typeof renderForm>["result"]) =>
  result.current.pickDirty(result.current.form.getValues());

describe("usePickDirty", () => {
  it("sends nothing when the user opens the form and saves without editing", () => {
    const { result } = renderForm();

    expect(patchOf(result)).toEqual({});
  });

  it("omits untouched keys entirely rather than sending them as undefined", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });

    const patch = patchOf(result);
    expect(patch).toEqual({ team_alias: "team-b" });
    expect("max_budget" in patch).toBe(false);
  });

  it("sends every edited field, not just the one that first made the form dirty", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });
    act(() => {
      result.current.form.setValue("models", [], { shouldDirty: true });
    });
    act(() => {
      result.current.form.setValue("max_budget", null, { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({ team_alias: "team-b", models: [], max_budget: null });
  });

  describe("clear tokens survive", () => {
    it("keeps null, which is how a scalar is cleared", () => {
      const { result } = renderForm();

      act(() => {
        result.current.form.setValue("max_budget", null, { shouldDirty: true });
      });

      const patch = patchOf(result);
      expect("max_budget" in patch).toBe(true);
      expect(patch.max_budget).toBeNull();
    });

    it("keeps an empty string", () => {
      const { result } = renderForm();

      act(() => {
        result.current.form.setValue("team_alias", "", { shouldDirty: true });
      });

      expect(patchOf(result)).toEqual({ team_alias: "" });
    });

    it("keeps zero", () => {
      const { result } = renderForm();

      act(() => {
        result.current.form.setValue("max_budget", 0, { shouldDirty: true });
      });

      expect(patchOf(result)).toEqual({ max_budget: 0 });
    });

    it("keeps an empty array, which is how lists are cleared", () => {
      const { result } = renderForm();

      act(() => {
        result.current.form.setValue("models", [], { shouldDirty: true });
      });

      expect(patchOf(result)).toEqual({ models: [] });
    });

    it("keeps an empty object, which is how model_aliases is cleared", () => {
      const { result } = renderForm();

      act(() => {
        result.current.form.setValue("model_aliases", {}, { shouldDirty: true });
      });

      expect(patchOf(result)).toEqual({ model_aliases: {} });
    });
  });

  it("sends an empty array to clear a list emptied through useFieldArray", () => {
    const { result } = renderForm();

    act(() => {
      result.current.fieldArray.remove(0);
    });

    expect(patchOf(result)).toEqual({ modelLimits: [] });
  });

  it("sends the whole array when one element of a field array changes", () => {
    const { result } = renderForm();

    act(() => {
      result.current.fieldArray.append({ model: "opus", tpm: 2 });
    });

    expect(patchOf(result)).toEqual({
      modelLimits: [
        { model: "gpt-4", tpm: 1 },
        { model: "opus", tpm: 2 },
      ],
    });
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

  it("drops a list the user edited and then reverted while another field stays dirty", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("team_alias", "team-b", { shouldDirty: true });
    });
    act(() => {
      result.current.form.setValue("models", ["gpt-4"], { shouldDirty: true });
    });
    act(() => {
      result.current.form.setValue("models", ["gpt-4", "opus"], { shouldDirty: true });
    });

    expect(patchOf(result)).toEqual({ team_alias: "team-b" });
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

  it("keeps value identity so nested references are not cloned", () => {
    const { result } = renderForm();

    act(() => {
      result.current.form.setValue("models", ["opus"], { shouldDirty: true });
    });

    const values = result.current.form.getValues();
    expect(result.current.pickDirty(values).models).toBe(values.models);
  });
});
