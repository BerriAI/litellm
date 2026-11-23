import type { ComponentProps, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import UsageStatStrip from "../../usage/_components/components/overview/UsageStatStrip";
import CustomerSurchargeBreakdown from "./CustomerSurchargeBreakdown";
import { customerUsagePresets, CUSTOMER_USAGE_MAX_RANGE_DAYS } from "./customerUsageRange";

export function CustomerUsageDatePicker(
  props: Pick<ComponentProps<typeof AdvancedDatePicker>, "value" | "onValueChange" | "presetShortLabel">,
) {
  const { t } = useTranslation();
  return (
    <AdvancedDatePicker
      {...props}
      presets={customerUsagePresets}
      maxRangeDays={CUSTOMER_USAGE_MAX_RANGE_DAYS}
      label={t("customerPortal.range.label")}
      showTimeRange={false}
      compact
      className="max-w-sm"
    />
  );
}

export function CustomerUsageStats({
  surcharge,
  children,
  ...props
}: Pick<ComponentProps<typeof UsageStatStrip>, "results" | "totals" | "loading"> & {
  surcharge?: number | null;
  children?: ReactNode;
}) {
  return (
    <UsageStatStrip
      {...props}
      requestCountsPending={props.loading}
      budget={null}
      showBudgetHint={false}
      spendDetails={
        <>
          <CustomerSurchargeBreakdown spend={props.totals.spend} surcharge={surcharge} />
          {children}
        </>
      }
    />
  );
}
