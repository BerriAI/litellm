import { z } from "zod";
import { budgetEditorFieldsSchema } from "../customerBudgetFieldsSchema";
import type { CustomerHierarchyEntity, Organization } from "../customerHierarchyForms";
import { toHierarchyPatch } from "./mapper";

export const hierarchyEditSchema = (entity: CustomerHierarchyEntity, organization: Organization) =>
  z
    .object({
      alias: z.string(),
      budget: budgetEditorFieldsSchema.extend({ templateId: z.string() }),
    })
    .transform((fields) => toHierarchyPatch(entity, organization, fields));
