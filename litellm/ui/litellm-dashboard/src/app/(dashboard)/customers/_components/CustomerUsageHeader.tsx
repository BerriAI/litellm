import type { ParseKeys } from "i18next";
import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { uiHref } from "@/utils/uiHref";
import type { CustomerHierarchyUsage, CustomerUsageScope } from "./customerUsageApi";

interface CustomerUsageHeaderProps {
  scope: CustomerUsageScope;
  data: CustomerHierarchyUsage | undefined;
  dateValue: DateRangePickerValue;
  preset: string | null;
  onOrganizationClick: () => void;
  onTeamClick: () => void;
  onDateRangeChange: (value: DateRangePickerValue, preset: string | null) => void;
}

const scopeLabel = (scope: CustomerUsageScope): ParseKeys => {
  if (scope.member) return "customerPortal.analytics.memberUsage";
  if (scope.team) return "customerPortal.analytics.teamUsage";
  return "customerPortal.analytics.organizationUsage";
};

function CustomerUsageBreadcrumbs({
  scope,
  data,
  onOrganizationClick,
  onTeamClick,
}: Pick<CustomerUsageHeaderProps, "scope" | "data" | "onOrganizationClick" | "onTeamClick">) {
  const { t } = useTranslation();
  return (
    <nav
      aria-label={t("customerPortal.analytics.breadcrumb")}
      className="mb-5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
    >
      <a href={uiHref("customers")} className="hover:text-foreground">
        {t("customerPortal.entities.organizations")}
      </a>
      <ChevronRight aria-hidden="true" className="size-3" />
      <button type="button" className="hover:text-foreground" onClick={onOrganizationClick}>
        {data?.organization.label ?? scope.org}
      </button>
      {scope.team && (
        <>
          <ChevronRight aria-hidden="true" className="size-3" />
          <button type="button" className="hover:text-foreground" onClick={onTeamClick}>
            {data?.team?.label ?? scope.team}
          </button>
        </>
      )}
      {scope.member && (
        <>
          <ChevronRight aria-hidden="true" className="size-3" />
          <span aria-current="page">{data?.member?.label ?? scope.member}</span>
        </>
      )}
    </nav>
  );
}

export default function CustomerUsageHeader({
  scope,
  data,
  dateValue,
  preset,
  onOrganizationClick,
  onTeamClick,
  onDateRangeChange,
}: CustomerUsageHeaderProps) {
  const { t } = useTranslation();
  const entity = data?.member ?? data?.team ?? data?.organization;
  const label = entity?.label ?? scope.member ?? scope.team ?? scope.org;
  return (
    <header>
      <CustomerUsageBreadcrumbs
        scope={scope}
        data={data}
        onOrganizationClick={onOrganizationClick}
        onTeamClick={onTeamClick}
      />
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-1 text-xs text-muted-foreground">{t(scopeLabel(scope))}</p>
          <h1 className="text-2xl font-semibold tracking-tight">{label}</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">{t("customerPortal.analytics.subtitle")}</p>
        </div>
        <AdvancedDatePicker
          value={dateValue}
          presetShortLabel={preset}
          showTimeRange={false}
          onValueChange={onDateRangeChange}
        />
      </div>
    </header>
  );
}
