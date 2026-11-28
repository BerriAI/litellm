import { budgetSelectionFields, budgetSelectionValue } from "../../customers/_components/customerBudgetForm";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import { appSurchargeField, appSurchargeValue, type App, type AppResource } from "./appForms";
import type { AppSettingsDraft } from "./appSettingsSchema";

export const appSettingsDraft = (
  app: App | undefined,
  profiles: BudgetGroups["default_budgets"],
  defaultBudgetId?: string | null,
): AppSettingsDraft => ({
  ...budgetSelectionFields(app, profiles, defaultBudgetId),
  autoCreate: app?.auto_create_customers ?? true,
  surcharge: appSurchargeField(app?.customer_margin),
  appGroupId: app?.app_group_id ?? null,
});

export const appSettingsBody = (fields: AppSettingsDraft, app?: App) => {
  const budget = budgetSelectionValue(fields, app?.budget);
  if (!budget.ok) return budget;
  const margin = appSurchargeValue(fields.surcharge);
  if (!margin.ok) return { ok: false, error: "apps.surchargeInvalid" } as const;
  return {
    ok: true,
    value: {
      ...budget.value,
      ...(fields.appGroupId !== (app?.app_group_id ?? null) ? { app_group_id: fields.appGroupId } : {}),
      auto_create_customers: fields.autoCreate,
      customer_margin: margin.value,
    },
  } as const;
};

export const appResourceBody = (resource: Pick<AppResource, "resource_id" | "resource_type">) => ({
  resource_id: resource.resource_id,
  resource_type: resource.resource_type,
});

export const createAppSettingsBody = (
  body: Extract<ReturnType<typeof appSettingsBody>, { ok: true }>["value"],
  resource: Pick<AppResource, "resource_id" | "resource_type">,
) => ({ ...body, ...appResourceBody(resource) });
