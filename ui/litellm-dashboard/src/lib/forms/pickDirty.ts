import type { Control, FieldValues, FormState } from "react-hook-form";
import { useFormState } from "react-hook-form";

const isDirtyNode = (node: unknown): boolean => {
  if (typeof node === "boolean") {
    return node;
  }
  if (Array.isArray(node)) {
    return node.some(isDirtyNode);
  }
  if (node !== null && typeof node === "object") {
    return Object.values(node as Record<string, unknown>).some(isDirtyNode);
  }
  return false;
};

const pickDirtyFields = <TValues extends FieldValues>(
  values: TValues,
  dirtyFields: FormState<TValues>["dirtyFields"],
): Partial<TValues> =>
  Object.fromEntries(
    Object.keys(values)
      .filter((key) => isDirtyNode((dirtyFields as Record<string, unknown>)[key]))
      .map((key) => [key, values[key]]),
  ) as Partial<TValues>;

// RHF only keeps dirtyFields current for subscribers, and formState.dirtyFields read inside a submit handler is a stale
// snapshot; reading it here during render is what turns the subscription on
export const usePickDirty = <TValues extends FieldValues>(
  control: Control<TValues>,
): ((values: TValues) => Partial<TValues>) => {
  const { dirtyFields } = useFormState({ control });
  return (values) => pickDirtyFields(values, dirtyFields);
};
