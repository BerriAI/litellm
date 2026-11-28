import type { TFunction } from "i18next";
import { z } from "zod";
import type { BudgetEditorFields } from "../../customers/_components/customerBudgetForm";
import type { AppOrganization, Budget } from "./appForms";
import { organizationAppBody } from "./organizationAppMapper";

export interface OrganizationAppDraft {
  source: string;
  budget: BudgetEditorFields;
  policy: Budget | null;
}

export const createOrganizationAppSchema = (original: Budget | null, row: AppOrganization | null, t: TFunction) =>
  z.object({ draft: z.custom<OrganizationAppDraft>() }).transform(({ draft }, context) => {
    const result = organizationAppBody(draft, original, row);
    if (result.ok) return result.value;
    context.addIssue({ code: "custom", message: t(result.error) });
    return z.NEVER;
  });
