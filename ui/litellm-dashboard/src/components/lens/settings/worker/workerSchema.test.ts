import { expect, it } from "vitest";
import { analysisAccessSchema, workerFormSchema } from "./workerSchema";

const workerDefaults = {
  useExisting: false,
  analysisKey: null,
  access: { model: null, budget: "100" },
};

it.each(["", "0", "-1", "NaN", "Infinity"])("rejects an invalid analysis budget: %s", (budget) => {
  const result = analysisAccessSchema.safeParse({ model: "analysis", budget });
  expect(result.success).toBe(false);
  if (!result.success)
    expect(result.error.issues[0].message).toBe("Choose a model and a monthly limit greater than zero");
});

it("requires a model and converts an accepted analysis budget to a number", () => {
  expect(analysisAccessSchema.safeParse({ model: null, budget: "100" }).success).toBe(false);
  expect(analysisAccessSchema.parse({ model: "analysis", budget: "0.01" })).toEqual({
    model: "analysis",
    budget: 0.01,
  });
});

it("requires a billing key when using existing access", () => {
  const result = workerFormSchema.safeParse({
    ...workerDefaults,
    useExisting: true,
  });
  expect(result.success).toBe(false);
  if (result.success) return;
  expect(result.error.issues.map(({ path }) => path)).toEqual([["analysisKey"]]);
});

it("validates analysis access only when creating a new virtual key", () => {
  const invalidAccess = workerFormSchema.safeParse(workerDefaults);
  expect(invalidAccess.success).toBe(false);
  if (!invalidAccess.success) expect(invalidAccess.error.issues[0].path).toEqual(["access"]);
  expect(
    workerFormSchema.safeParse({
      ...workerDefaults,
      access: { model: "analysis", budget: "100" },
    }).success,
  ).toBe(true);
  expect(
    workerFormSchema.safeParse({
      ...workerDefaults,
      useExisting: true,
      analysisKey: "key",
    }).success,
  ).toBe(true);
});
