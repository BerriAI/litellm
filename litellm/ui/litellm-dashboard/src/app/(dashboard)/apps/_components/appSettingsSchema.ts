import type { TFunction } from "i18next";
import { z } from "zod";
import type { budgetSelectionFields } from "../../customers/_components/customerBudgetForm";
import type { App } from "./appForms";
import { appSettingsBody } from "./appSettingsMapper";

export type AppSettingsDraft = ReturnType<typeof budgetSelectionFields> & {
  autoCreate: boolean;
  surcharge: string;
  appGroupId: string | null;
};

export const createAppSettingsSchema = (app: App | undefined, t: TFunction) =>
  z.object({ draft: z.custom<AppSettingsDraft>() }).transform(({ draft }, context) => {
    const result = appSettingsBody(draft, app);
    if (result.ok) return result.value;
    context.addIssue({ code: "custom", message: t(result.error) });
    return z.NEVER;
  });
