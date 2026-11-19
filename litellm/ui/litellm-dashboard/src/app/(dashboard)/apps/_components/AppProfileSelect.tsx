"use client";

import { ErrorState } from "@/components/shared/ErrorState";
import { useTranslation } from "react-i18next";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import { CustomerBudgetSourceSelect } from "../../customers/_components/CustomerLimitsFields";
import { useAppBudgetGroups } from "./useAppBudgetGroups";

export default function AppProfileSelect({
  value,
  onChange,
  emptyLabel,
}: {
  value: string;
  onChange: (value: string) => void;
  emptyLabel: string;
}) {
  const { t } = useTranslation();
  const groups = useAppBudgetGroups([value]);
  return (
    <div className="space-y-4">
      <CustomerBudgetSourceSelect
        profiles={groups.data?.default_budgets ?? []}
        title={t("apps.profile")}
        value={value ? `profile:${value}` : "inherit"}
        onChange={(value) => onChange(value?.startsWith("profile:") ? value.slice("profile:".length) : "")}
        disabled={!groups.isSuccess}
        inheritedOption={{ label: emptyLabel, summary: t("apps.noAdditionalLimits") }}
        custom={false}
      />
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
