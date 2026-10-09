import { describe, expectTypeOf, test } from "vitest";
import type { FieldPath } from "react-hook-form";
import type { InvestigationField, InvestigationInput, SetupStep } from "./investigationSchema";
import { investigationStepFields } from "./investigationSchema";

describe("investigation step fields", () => {
  test("every step field is a path react-hook-form can validate", () => {
    expectTypeOf<InvestigationField>().toExtend<FieldPath<InvestigationInput>>();
  });

  test("step fields are the form's top-level fields with selection expanded one level", () => {
    expectTypeOf<InvestigationField>().toEqualTypeOf<
      Exclude<keyof InvestigationInput, "selection"> | `selection.${keyof InvestigationInput["selection"]}`
    >();
  });

  test("a step that is not in the stepper has no field list", () => {
    expectTypeOf(investigationStepFields).toHaveProperty("activity");
    expectTypeOf<keyof typeof investigationStepFields>().toEqualTypeOf<SetupStep>();
    // @ts-expect-error summary is not a setup step
    expectTypeOf(investigationStepFields.summary);
  });
});
