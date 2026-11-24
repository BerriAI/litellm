"use client";

import { useQueryClient } from "@tanstack/react-query";
import { createSerializer } from "nuqs";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useAppGroups } from "./appGroupApi";
import { fetchClient } from "@/lib/http/api";
import { uiHref } from "@/utils/uiHref";
import {
  budgetSelectionFields,
  budgetSelectionValue,
  changeBudgetSelection,
} from "../../customers/_components/customerBudgetForm";
import { useCustomerBudgetForm } from "../../customers/_components/useCustomerBudgetForm";
import { CustomerBudgetForm } from "../../customers/_components/CustomerBudgetDialog";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import { appSurchargeField, appSurchargeValue, type App, type AppResource } from "./appForms";
import { invalidateAppBudgets } from "./appApi";
import { CustomerBudgetEditor } from "../../customers/_components/CustomerLimitsFields";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";

const appHref = createSerializer(customerAppParsers);

const appDefaultsFields = (
  app: App | undefined,
  profiles: BudgetGroups["default_budgets"],
  defaultBudgetId?: string | null,
) => {
  return {
    ...budgetSelectionFields(app, profiles, defaultBudgetId),
    autoCreate: app?.auto_create_customers ?? true,
    surcharge: appSurchargeField(app?.customer_margin),
    appGroupId: app?.app_group_id ?? null,
  };
};

export default function AppSettingsForm({
  resource,
  app,
  profiles,
  defaultBudgetId,
  onClose,
  onPendingChange,
}: {
  resource: AppResource;
  app?: App;
  profiles: BudgetGroups["default_budgets"];
  defaultBudgetId?: string | null;
  onClose?: () => void;
  onPendingChange?: (pending: boolean) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const appGroups = useAppGroups();
  const { fields, setFields, update, pending, error, save, dirty } = useCustomerBudgetForm(
    {
      initial: appDefaultsFields(app, profiles, defaultBudgetId),
      parse: (fields) => {
        const budget = budgetSelectionValue(fields, app?.budget);
        if (!budget.ok) return { ok: false, error: t(budget.error) };
        const margin = appSurchargeValue(fields.surcharge);
        if (!margin.ok) return { ok: false, error: t("apps.surchargeInvalid") };
        return {
          ok: true,
          value: {
            ...budget.value,
            ...(fields.appGroupId !== (app?.app_group_id ?? null) ? { app_group_id: fields.appGroupId } : {}),
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
    setFields((current) => ({ ...current, ...changeBudgetSelection(current, source, profiles) }));
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
      <div className="space-y-2">
        <label htmlFor="app-parent-group" className="text-sm font-medium">
          {t("apps.groups.parent")}
        </label>
        <SearchSelect
          id="app-parent-group"
          className="w-full"
          value={fields.appGroupId ?? undefined}
          placeholder={t("apps.groups.noGroup")}
          onValueChange={(value) => update("appGroupId", value)}
          disabled={pending || appGroups.isPending || appGroups.isError}
          options={(appGroups.data ?? []).map((group) => ({ value: group.app_group_id, label: group.name }))}
        />
        <p className="text-xs text-muted-foreground">{t("apps.groups.membershipHint")}</p>
        {appGroups.isError && (
          <ErrorState message={extractProxyErrorMessage(appGroups.error)} onRetry={() => void appGroups.refetch()} />
        )}
      </div>
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
