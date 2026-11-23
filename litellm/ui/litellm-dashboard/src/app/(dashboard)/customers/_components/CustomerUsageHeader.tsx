import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import CustomerHierarchyBreadcrumbs from "./CustomerHierarchyBreadcrumbs";
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
        <CustomerHierarchyBreadcrumbs
          scope={{ ...scope, team: data?.team?.id ?? scope.team }}
          page="usage"
          organizationLabel={data?.organization.label}
          teamLabel={data?.team?.label}
          memberLabel={data?.member?.label}
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
