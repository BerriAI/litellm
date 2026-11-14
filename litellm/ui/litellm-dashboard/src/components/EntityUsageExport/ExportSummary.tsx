import React from "react";
import { useTranslation } from "react-i18next";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";

interface ExportSummaryProps {
  dateRange: DateRangePickerValue;
  selectedFilters: readonly string[];
}

const ExportSummary: React.FC<ExportSummaryProps> = ({ dateRange, selectedFilters }) => {
  const { t } = useTranslation();
  return (
    <div className="text-sm text-muted-foreground">
      {dateRange.from?.toLocaleDateString()} - {dateRange.to?.toLocaleDateString()}
      {selectedFilters.length > 0 && ` · ${t("usage.export.summary.filterCount", { count: selectedFilters.length })}`}
    </div>
  );
};

export default ExportSummary;
