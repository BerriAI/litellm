import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { useCustomerHierarchyLinks } from "./CustomerHierarchyNavigation";
import { CustomerUsageDatePicker } from "./CustomerUsageSummary";
import type { CustomerHierarchyUsage, CustomerUsageScope } from "./customerUsageApi";
import { CustomerDetailHeader } from "./CustomerDetailLayout";

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
    <>
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
    </>
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
  const entity = data?.member ?? data?.team ?? data?.organization;
  const label = entity?.label ?? scope.member ?? scope.team ?? scope.org;
  return (
    <CustomerDetailHeader
      title={label}
      breadcrumbs={
        <CustomerUsageBreadcrumbs
          scope={scope}
          data={data}
          onOrganizationClick={onOrganizationClick}
          onTeamClick={onTeamClick}
        />
      }
      actions={
        <CustomerUsageDatePicker value={dateValue} presetShortLabel={preset} onValueChange={onDateRangeChange} />
      }
    />
  );
}
