import { z } from "zod";
import { budgetEditorFieldsSchema } from "../customerBudgetFieldsSchema";
import type { CustomerHierarchyCreateTarget } from "../customerHierarchyForms";
import { toHierarchyCreate } from "./mapper";

export const hierarchyCreateSchema = (target: CustomerHierarchyCreateTarget) =>
  z
    .object({
      alias: z.string(),
      identifier: z.string(),
      budget: budgetEditorFieldsSchema.extend({ templateId: z.string() }),
    })
    .transform((fields) => toHierarchyCreate(target, fields));
