"use client";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { useTranslation } from "react-i18next";
import { $api } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import AppBudgetLimits from "./AppBudgetLimits";

export default function AppProfileSelect({
  id,
  value,
  onChange,
  emptyLabel,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  emptyLabel: string;
}) {
  const { t } = useTranslation();
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {});
  const selected = groups.data?.default_budgets.find((group) => group.budget_id === value);
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-sm">
        {t("apps.profile")}
      </label>
      <SearchSelect
        id={id}
        className="w-full"
        value={value}
        placeholder={emptyLabel}
        onValueChange={(value) => onChange(value ?? "")}
        disabled={!groups.data}
        options={[
          { value: "", label: emptyLabel },
          ...(groups.data?.default_budgets ?? []).map((group) => ({ value: group.budget_id, label: group.name })),
        ]}
      />
      {selected && <AppBudgetLimits budget={selected} />}
      <p className="text-xs text-muted-foreground">
        <a href={uiHref("customers/settings")} className="text-primary hover:underline">
          {t("apps.manageProfiles")}
        </a>
      </p>
      {groups.isError && (
        <p role="alert" className="text-sm text-destructive">
          {extractProxyErrorMessage(groups.error)}
        </p>
      )}
    </div>
  );
}
