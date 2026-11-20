"use client";

import { useQueryClient } from "@tanstack/react-query";
import { createSerializer } from "nuqs";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { fetchClient } from "@/lib/http/api";
import { uiHref } from "@/utils/uiHref";
import {
  budgetEditorFields,
  budgetPolicyChanges,
  permanentBudgetEditorValue,
} from "../../customers/_components/customerBudgetForm";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import { appSurchargeField, appSurchargeValue, type App, type AppResource } from "./appForms";
import { invalidateAppBudgets } from "./appApi";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";

const appHref = createSerializer(customerAppParsers);

const appDefaultsFields = (app: App | undefined, profiles: BudgetGroups["default_budgets"]) => {
  const customSource = app?.budget ? "custom" : "inherit";
  return {
    budget: budgetEditorFields(app?.budget),
    policy: app?.budget,
    budgetSource: profiles.some((profile) => profile.budget_id === app?.budget_id)
      ? `profile:${app?.budget_id}`
      : customSource,
    autoCreate: app?.auto_create_customers ?? true,
    surcharge: appSurchargeField(app?.customer_margin),
  };
};

export default function AppSettingsForm({
  resource,
  app,
  profiles,
  onClose,
  onPendingChange,
}: {
  resource: AppResource;
  app?: App;
  profiles: BudgetGroups["default_budgets"];
  onClose?: () => void;
  onPendingChange?: (pending: boolean) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { fields, setFields, update, pending, error, save, dirty } = useCustomerBudgetForm(
    {
      initial: appDefaultsFields(app, profiles),
      parse: (fields) => {
        const budget =
          fields.budgetSource === "custom"
            ? permanentBudgetEditorValue(fields.budget, app?.budget)
            : ({ ok: true, value: null } as const);
        if (!budget.ok) return { ok: false, error: t(budget.error) };
        const margin = appSurchargeValue(fields.surcharge);
        if (!margin.ok) return { ok: false, error: t("apps.surchargeInvalid") };
        return {
          ok: true,
          value: {
            ...(fields.budgetSource === "custom"
              ? { budget: { ...budget.value, ...budgetPolicyChanges(fields.policy, app?.budget) } }
              : { budget_id: fields.budgetSource.startsWith("profile:") ? fields.budgetSource.slice(8) : null }),
            auto_create_customers: fields.autoCreate,
            customer_margin: margin.value,
          },
        };
      },
      write: async (body) => {
        const response = app
          ? await fetchClient.PATCH("/management/v1/apps/{app_id}", { params: { path: { app_id: app.app_id } }, body })
          : await fetchClient.POST("/management/v1/apps", {
              body: { ...body, resource_id: resource.resource_id, resource_type: resource.resource_type },
            });
        if (response.data) setFields(appDefaultsFields(response.data, profiles));
        await invalidateAppBudgets(queryClient);
        toast.success(t("apps.saved"));
        onClose?.();
      },
    },
    onPendingChange,
  );
  const changeBudgetSource = (source: string | null) => {
    if (!source) return;
    setFields((current) => {
      const selected =
        source === "custom"
          ? profiles.find((profile) => `profile:${profile.budget_id}` === current.budgetSource)
          : null;
      return {
        ...current,
        budgetSource: source,
        budget: { ...(selected ? budgetEditorFields(selected) : current.budget), budgetEnabled: source === "custom" },
        policy: selected ?? current.policy,
      };
    });
  };
  return (
    <CustomerBudgetForm
      layout={onClose ? "dialog" : "page"}
      pending={pending}
      dirty={onClose ? true : dirty}
      error={error}
      onSubmit={save}
      onClose={onClose}
      saveLabel={t("apps.save")}
      label={t("apps.defaultsFor", { alias: resource.alias })}
    >
      <CustomerBudgetEditor
        title={t("apps.appBudget")}
        description={t("apps.appBudgetHint")}
        id="app-budget"
        budget={{
          fields: fields.budget,
          onChange: (value) => update("budget", value),
          disabled: pending,
          modelScope: resource,
        }}
        source={{
          profiles,
          value: fields.budgetSource,
          onChange: changeBudgetSource,
          title: t("apps.appBudget"),
          disabled: pending,
          inheritedOption: {
            label: t("customerPortal.createDialog.noLimit"),
            summary: t("apps.noAdditionalLimits"),
          },
        }}
      />
      <a href={`${uiHref("customers/settings")}?tab=overall-app`} className="text-xs text-primary hover:underline">
        {t("apps.manageOverallProfiles")}
      </a>
      {onClose && app && (
        <div className="border-t pt-6">
          <a
            href={appHref(uiHref("apps/budget-groups"), { app_id: app.app_id })}
            className="text-sm text-primary hover:underline"
          >
            {t("apps.manageMemberProfiles")}
          </a>
        </div>
      )}
      <div className="space-y-1.5 border-t pt-4">
        <label className="flex items-center gap-2 text-sm font-medium">
          <Checkbox
            checked={fields.autoCreate}
            disabled={pending}
            onCheckedChange={(checked) => update("autoCreate", checked === true)}
          />
          {t("apps.autoCreateCustomers")}
        </label>
        <p className="text-xs text-muted-foreground">{t("apps.autoCreateCustomersHint")}</p>
      </div>
      <div className="max-w-md space-y-1.5 border-t pt-4">
        <label htmlFor="app-surcharge" className="block text-sm font-medium">
          {t("apps.surcharge")}
        </label>
        <Input
          id="app-surcharge"
          type="number"
          step="any"
          className="w-full"
          placeholder={t("apps.surchargePlaceholder")}
          disabled={pending}
          value={fields.surcharge}
          onChange={(event) => update("surcharge", event.target.value)}
        />
        <p className="text-xs text-muted-foreground">{t("apps.surchargeHint")}</p>
      </div>
    </CustomerBudgetForm>
  );
}
