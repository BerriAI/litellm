import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { useCustomerHierarchyLinks } from "./CustomerHierarchyNavigation";
import { customerUsagePresets, CUSTOMER_USAGE_MAX_RANGE_DAYS } from "./customerUsageRange";
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

function CustomerUsageBreadcrumbs({
  scope,
  data,
  onOrganizationClick,
  onTeamClick,
}: Pick<CustomerUsageHeaderProps, "scope" | "data" | "onOrganizationClick" | "onTeamClick">) {
  const { t } = useTranslation();
  const { customersHref } = useCustomerHierarchyLinks();
  return (
    <nav
      aria-label={t("customerPortal.analytics.breadcrumb")}
      className="mb-5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
    >
      <a href={customersHref} className="hover:text-foreground">
        {t("customerPortal.analytics.allCustomers")}
      </a>
      {(scope.team || scope.member) && (
        <>
          <ChevronRight aria-hidden="true" className="size-3" />
          <button type="button" className="hover:text-foreground" onClick={onOrganizationClick}>
            {data?.organization.label ?? scope.org}
          </button>
        </>
      )}
      {scope.member && scope.team && (
        <>
          <ChevronRight aria-hidden="true" className="size-3" />
          <button type="button" className="hover:text-foreground" onClick={onTeamClick}>
            {data?.team?.label ?? scope.team}
          </button>
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
        <h1 className="min-w-0 break-words text-2xl font-semibold tracking-tight">{label}</h1>
        <AdvancedDatePicker
          value={dateValue}
          presetShortLabel={preset}
          label={t("customerPortal.range.label")}
          showTimeRange={false}
          presets={customerUsagePresets}
          maxRangeDays={CUSTOMER_USAGE_MAX_RANGE_DAYS}
          onValueChange={onDateRangeChange}
          compact
          className="max-w-sm"
        />
      </div>
    </header>
  );
}
