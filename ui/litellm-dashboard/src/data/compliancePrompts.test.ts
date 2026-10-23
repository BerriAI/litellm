import { describe, expect, it } from "vitest";
import { toSiTerminology } from "@/lib/terminology";
import { getComplianceDatasetPrompts, getFrameworks } from "./compliancePrompts";

describe("getFrameworks with a terminology", () => {
  const original = getFrameworks();
  const si = getFrameworks(toSiTerminology);
  const categoriesOf = (frameworks: typeof original) => frameworks.flatMap((framework) => framework.categories);
  const promptsOf = (frameworks: typeof original) => categoriesOf(frameworks).flatMap((category) => category.prompts);

  it("rewrites every prompt the playground sends to the model", () => {
    expect(promptsOf(si).length).toBeGreaterThan(0);
    expect(promptsOf(si).map((prompt) => prompt.prompt)).toEqual(
      promptsOf(original).map((prompt) => toSiTerminology(prompt.prompt)),
    );
    expect(promptsOf(si).some((prompt, index) => prompt.prompt !== promptsOf(original)[index].prompt)).toBe(true);
  });

  it("rewrites framework and category descriptions", () => {
    expect(si.map((framework) => framework.description)).toEqual(
      original.map((framework) => toSiTerminology(framework.description)),
    );
    expect(categoriesOf(si).map((category) => category.description)).toEqual(
      categoriesOf(original).map((category) => toSiTerminology(category.description)),
    );
  });

  it("keeps framework names, category names and prompt ids stable for selection and CSV export", () => {
    expect(si.map((framework) => framework.name)).toEqual(original.map((framework) => framework.name));
    expect(categoriesOf(si).map((category) => category.name)).toEqual(
      categoriesOf(original).map((category) => category.name),
    );
    expect(promptsOf(si).map((prompt) => prompt.id)).toEqual(promptsOf(original).map((prompt) => prompt.id));
  });

  it("feeds the rewritten prompts into the flat dataset used by the pipeline tester", () => {
    expect(getComplianceDatasetPrompts(toSiTerminology)).toEqual(promptsOf(si));
  });
});
