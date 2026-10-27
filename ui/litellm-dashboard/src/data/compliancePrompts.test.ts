import { describe, expect, it } from "vitest";
import { toSiTerminology } from "@/lib/terminology";
import { getFrameworks } from "./compliancePrompts";

describe("getFrameworks with a terminology", () => {
  const original = getFrameworks();
  const si = getFrameworks(toSiTerminology);
  const categoriesOf = (frameworks: typeof original) => frameworks.flatMap((framework) => framework.categories);
  const promptsOf = (frameworks: typeof original) => categoriesOf(frameworks).flatMap((category) => category.prompts);

  it("keeps the prompts the playground sends to the model in their original wording", () => {
    expect(promptsOf(si).length).toBeGreaterThan(0);
    expect(promptsOf(original).some((prompt) => toSiTerminology(prompt.prompt) !== prompt.prompt)).toBe(true);
    expect(promptsOf(si).map((prompt) => prompt.prompt)).toEqual(promptsOf(original).map((prompt) => prompt.prompt));
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
});
