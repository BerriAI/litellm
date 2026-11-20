"use client";

import { SearchSelect } from "@/components/shared/SearchSelect";
import { ErrorState } from "@/components/shared/ErrorState";
import { useTranslation } from "react-i18next";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import AppBudgetLimits from "./AppBudgetLimits";
import { useAppBudgetGroups } from "./useAppBudgetGroups";

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
  const groups = useAppBudgetGroups([value]);
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
        disabled={!groups.isSuccess}
        options={[
          { value: "", label: emptyLabel },
          ...(groups.data?.default_budgets ?? []).map((group) => ({ value: group.budget_id, label: group.name })),
        ]}
      />
      {selected && <AppBudgetLimits budget={selected} />}
      <p className="text-xs text-muted-foreground">
        <a href={`${uiHref("customers/settings")}?tab=app`} className="text-primary hover:underline">
          {t("apps.manageProfiles")}
        </a>
      </p>
      {groups.isError && (
        <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
      )}
    </div>
  );
}
