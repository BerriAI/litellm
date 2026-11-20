import type { FieldValues, FormState } from "react-hook-form";

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

export const pickDirty = <TValues extends FieldValues>(
  values: TValues,
  dirtyFields: FormState<TValues>["dirtyFields"],
): Partial<TValues> =>
  Object.fromEntries(
    Object.keys(values)
      .filter((key) => isDirtyNode((dirtyFields as Record<string, unknown>)[key]))
      .map((key) => [key, values[key]]),
  ) as Partial<TValues>;
