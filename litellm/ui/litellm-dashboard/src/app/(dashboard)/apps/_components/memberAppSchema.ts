import type { TFunction } from "i18next";
import { z } from "zod";
import type { BudgetEditorFields } from "../../customers/_components/customerBudgetForm";
import type { AppMember, Budget } from "./appForms";
import { memberAppBody } from "./memberAppMapper";

export const createMemberAppSchema = (original: Budget | null, row: AppMember, t: TFunction) =>
  z.object({ draft: z.custom<BudgetEditorFields>() }).transform(({ draft }, context) => {
    const result = memberAppBody(draft, original, row);
    if (result.ok) return result.value;
    context.addIssue({ code: "custom", message: t(result.error) });
    return z.NEVER;
  });
