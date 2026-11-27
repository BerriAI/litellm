import { z } from "zod";
import { budgetEditorFieldsSchema, budgetLimitFieldsSchema } from "../customerBudgetFieldsSchema";
import { toCustomerCreateBody } from "./mapper";

const customerCreateShape = {
  name: z.string(),
  customerOrgId: z.string(),
  limits: budgetLimitFieldsSchema.extend({
    budgetMode: z.enum(["inherit", "custom", "profile"]),
    budgetId: z.string(),
    budgetAmount: z.string(),
  }),
  budgetGroups: z.object({
    defaultGroupId: z.string(),
    groups: z.array(
      budgetEditorFieldsSchema.extend({
        id: z.string(),
        name: z.string(),
        capPolicy: z.enum(["pause_until_reset", "pause_allow_increase"]),
      }),
    ),
  }),
};

export const customerCreateSchema = z.object(customerCreateShape).transform(toCustomerCreateBody);
